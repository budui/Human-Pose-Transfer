import torch
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar, CustomPeriodicEvent
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
from ignite.utils import convert_tensor

from helper.custom_ignite_handlers.tensorboard_logger import TensorboardLogger, OptimizerParamsHandler, \
    OutputHandler, WeightsHistHandler
from helper.misc import make_2d_grid, custom_global_step_transform
from .data import get_data_loader, get_val_data_pairs
from .loss import MaskL1Loss
from .model import Generator2, Generator1, Discriminator


def get_trainer(config, device=torch.device("cuda")):
    cfg = config["model"]["generator1"]
    generator1 = Generator1(3 + 18, cfg["num_repeat"], cfg["middle_features_dim"],
                            cfg["channels_base"], cfg["image_size"])
    generator1.to(device)
    generator1.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

    cfg = config["model"]["generator2"]
    generator2 = Generator2(3 + 3, cfg["channels_base"], cfg["num_repeat"], cfg["num_skip_out_connect"], weight_init_way=cfg["weight_init_way"])
    generator2.to(device)
    print(generator2)

    discriminator = Discriminator(weight_init_way=config["model"]["discriminator"]["weight_init_way"])
    discriminator.to(device)
    print(discriminator)

    cfg = config["train"]["generator2"]
    generator2_optimizer = optim.Adam(generator2.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    cfg = config["train"]["discriminator"]
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    mask_l1_loss = MaskL1Loss(config["loss"]["mask_l1"]["mask_ratio"])
    mask_l1_loss.to(device)
    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    adversarial_loss.to(device)

    real_labels = torch.ones((config["train"]["batch_size"], 1), device=device)
    fake_labels = torch.zeros((config["train"]["batch_size"], 1), device=device)

    def _step(engine, batch):
        batch = convert_tensor(batch, device)
        with torch.no_grad():
            generated_img_1 = generator1(batch["condition_img"], batch["target_bone"])
        generated_img = generated_img_1 + generator2(batch["condition_img"], generated_img_1)

        generator2_optimizer.zero_grad()
        g2_gan_loss = adversarial_loss(discriminator(generated_img), real_labels)
        g2_mask_l1_loss = mask_l1_loss(generated_img, batch["target_img"], batch["target_mask"])
        g2_loss = config["loss"]["mask_l1"]["weight"] * g2_mask_l1_loss + \
                  config["loss"]["gan"]["weight"] * g2_gan_loss
        g2_loss.backward()
        generator2_optimizer.step()

        discriminator_optimizer.zero_grad()
        d_real_loss = adversarial_loss(discriminator(batch["target_img"]), real_labels)
        d_fake_loss = adversarial_loss(discriminator(generated_img.detach()), fake_labels)
        d_loss_1 = (d_fake_loss + d_real_loss) / 2

        d_real_loss = adversarial_loss(discriminator(batch["target_img"]), real_labels)
        d_fake_loss = adversarial_loss(discriminator(batch["condition_img"]), fake_labels)
        d_loss_2 = (d_fake_loss + d_real_loss) / 2

        d_loss = (d_loss_1 + d_loss_2) / 2
        d_loss.backward()
        discriminator_optimizer.step()

        return {
            "loss": {
                "g2_mask_l1_loss": g2_mask_l1_loss.item(),
                "g2_gan_loss": g2_gan_loss.item(),
                "g2_loss": g2_loss.item(),
                "d_loss": d_loss.item(),
                "d_loss_1": d_loss_1.item(),
                "d_loss_2": d_loss_2.item(),
            },
            "img": {
                "mask_img": batch["target_mask"].detach(),
                "condition_img": batch["condition_img"].detach(),
                "target_img": batch["target_img"].detach(),
                "generated_img_1": generated_img_1.detach(),
                "generated_img": generated_img.detach(),
            }
        }

    trainer = Engine(_step)

    RunningAverage(output_transform=lambda x: x["loss"]['g2_mask_l1_loss']).attach(trainer, 'g2_mask_l1_loss')
    RunningAverage(output_transform=lambda x: x["loss"]['g2_gan_loss']).attach(trainer, 'g2_gan_loss')
    RunningAverage(output_transform=lambda x: x["loss"]['g2_loss']).attach(trainer, 'g2_loss')
    RunningAverage(output_transform=lambda x: x["loss"]['d_loss_1']).attach(trainer, 'd_loss_1')
    RunningAverage(output_transform=lambda x: x["loss"]['d_loss']).attach(trainer, 'd_loss')
    RunningAverage(output_transform=lambda x: x["loss"]['d_loss_2']).attach(trainer, 'd_loss_2')

    ProgressBar(ncols=0).attach(trainer, ["g2_loss", "d_loss"])

    mcp = ModelCheckpoint(config["output"], "network", save_interval=config["log"]["model_checkpoint"]["save_interval"],
                          n_saved=config["log"]["model_checkpoint"]["n_saved"], require_empty=False,
                          save_as_state_dict=True, create_dir=True)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, mcp, to_save={"G2": generator2, "D": discriminator})

    check_cpe = CustomPeriodicEvent(n_iterations=config["log"]["check_freq"])
    check_cpe.attach(trainer)
    CHECK_EVENT = getattr(check_cpe.Events, "ITERATIONS_{}_COMPLETED".format(config["log"]["check_freq"]))

    loss_cpe = CustomPeriodicEvent(n_iterations=config["log"]["loss_freq"])
    loss_cpe.attach(trainer)
    LOSS_EVENT = getattr(loss_cpe.Events, "ITERATIONS_{}_COMPLETED".format(config["log"]["loss_freq"]))

    tb_logger = TensorboardLogger(config["output"])
    tb_writer = tb_logger.writer

    loss_gst = custom_global_step_transform(config["log"]["loss_freq"])
    check_gst = custom_global_step_transform(config["log"]["check_freq"])

    check_handlers = [
        (OutputHandler(tag="G2", metric_names=["g2_mask_l1_loss", "g2_gan_loss", "g2_loss"],
                       global_step_transform=loss_gst), LOSS_EVENT),
        (OutputHandler(tag="D", metric_names=["d_loss_1", "d_loss_2", "d_loss"],
                       global_step_transform=loss_gst), LOSS_EVENT),
        (OptimizerParamsHandler(discriminator_optimizer, param_name="lr", tag="D", global_step_transform=check_gst), CHECK_EVENT),
        (OptimizerParamsHandler(generator2_optimizer, param_name="lr", tag="G2", global_step_transform=check_gst), CHECK_EVENT),
        (WeightsHistHandler(generator2, tag="G2", global_step_transform=check_gst), CHECK_EVENT),
        (WeightsHistHandler(discriminator, tag="D", global_step_transform=check_gst), CHECK_EVENT),
    ]

    for ch, e in check_handlers:
        tb_logger.attach(trainer, log_handler=ch, event_name=e)

    val_data_pair = get_val_data_pairs(config)
    val_data_pair = convert_tensor(val_data_pair, device)

    @trainer.on(CHECK_EVENT)
    def log(engine):
        # from python3.7 dict will keep order so that .values() will result in same output
        tb_writer.add_image('Train/image', make_2d_grid(engine.state.output["img"].values()), engine.state.iteration)
        with torch.no_grad():
            generator1.eval()
            generator2.eval()
            generated_img_1 = generator1(val_data_pair["condition_img"], val_data_pair["target_bone"])
            generated_img = generator2(val_data_pair["condition_img"], generated_img_1) + generated_img_1
            output_imgs = [val_data_pair["target_mask"], val_data_pair["condition_img"],
                           val_data_pair["target_img"], generated_img_1, generated_img]
            tb_writer.add_image('Test/image', make_2d_grid(output_imgs), engine.state.iteration)
            generator1.train()
            generator2.train()

    return trainer


def run(config):
    train_data_loader = get_data_loader(config)
    trainer = get_trainer(config)
    trainer.run(train_data_loader, max_epochs=config["train"]["num_epoch"])
