import pytest

from pytorch_lightning.loggers import TensorBoardLogger


@pytest.fixture
def tensorboard_logger(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    tb_logger = TensorBoardLogger(save_dir=str(log_dir))
    yield tb_logger