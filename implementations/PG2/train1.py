import torch
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar, CustomPeriodicEvent
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
from ignite.utils import convert_tensor

from helper.custom_ignite_handlers.tensorboard_logger import TensorboardLogger, OutputHandler, WeightsHistHandler
from helper.misc import make_2d_grid, custom_global_step_transform
from .data import get_data_loader, get_val_data_pairs
from .loss import MaskL1Loss
from .model import Generator1


def get_trainer(config, device=torch.device("cuda")):
    cfg = config["model"]["generator1"]
    generator1 = Generator1(3 + 18, cfg["num_repeat"], cfg["middle_features_dim"],
                            cfg["channels_base"], cfg["image_size"])
    generator1.to(device)
    print(generator1)

    cfg = config["train"]["generator1"]
    generator1_optimizer = optim.Adam(generator1.parameters(), lr=cfg["lr"],
                                      betas=(cfg["beta1"], cfg["beta2"]))

    mask_l1_loss = MaskL1Loss(config["loss"]["mask_l1"]["mask_ratio"])
    mask_l1_loss.to(device)

    def _step(engine, batch):
        batch = convert_tensor(batch, device)
        generated_img = generator1(batch["condition_img"], batch["target_bone"])

        generator1_optimizer.zero_grad()
        loss = mask_l1_loss(generated_img, batch["target_img"], batch["target_mask"])
        loss.backward()
        generator1_optimizer.step()
        return {
            "loss": {
                "mask_l1": loss.item()
            },
            "img": {
                "mask_img": batch["target_mask"].detach(),
                "condition_img": batch["condition_img"].detach(),
                "target_img": batch["target_img"].detach(),
                "generated_img": generated_img.detach(),
            }
        }

    trainer = Engine(_step)

    RunningAverage(output_transform=lambda x: x["loss"]['mask_l1']).attach(trainer, 'mask_l1_loss')
    ProgressBar(ncols=0).attach(trainer, ["mask_l1_loss"])

    mcp = ModelCheckpoint(config["output"], "network", save_interval=config["log"]["model_checkpoint"]["save_interval"],
                          n_saved=config["log"]["model_checkpoint"]["n_saved"], require_empty=False,
                          save_as_state_dict=True, create_dir=True)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, mcp, to_save={"G1": generator1})

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
        (OutputHandler(tag="G1", metric_names=["mask_l1_loss"], global_step_transform=loss_gst), LOSS_EVENT),
        (WeightsHistHandler(generator1, tag="G1", global_step_transform=check_gst), CHECK_EVENT),
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
            generated_img_1 = generator1(val_data_pair["condition_img"], val_data_pair["target_bone"])
            output_imgs = [val_data_pair["target_mask"], val_data_pair["condition_img"],
                           val_data_pair["target_img"], generated_img_1]
            tb_writer.add_image('Test/image', make_2d_grid(output_imgs), engine.state.iteration)
            generator1.train()
    return trainer


def run(config):
    train_data_loader = get_data_loader(config)
    trainer = get_trainer(config)
    trainer.run(train_data_loader, max_epochs=config["train"]["num_epoch"])
