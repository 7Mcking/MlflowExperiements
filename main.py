import torch
import flash
import mlflow

from flash.core.data.utils import download_data

from flash.text import TextClassificationData, TextClassifier


download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")


def trainer(INPUT_FIELD, TARGET_FIELD, TRAIN_FILE, VAL_FILE, TEST_FILE, BATCH_SIZE):
    datamodule = TextClassificationData.from_csv(input_field=INPUT_FIELD,
                                                 target_fields=TARGET_FIELD,
                                                 train_file=TRAIN_FILE,
                                                 val_file=VAL_FILE,
                                                 test_file=TEST_FILE,
                                                 predict_file=PREDICT_FILE,
                                                 batch_size=BATCH_SIZE)

    # Build the model
    model = TextClassifier(backbone="prajjwal1/bert-tiny", num_classes=datamodule.num_classes, labels= datamodule.labels)

    # Define the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())

    # Fine-tune the model
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    trainer.validate(datamodule=datamodule)
    trainer.test(datamodule=datamodule)

    trainer.save_checkpoint("text_classification_model.pt")
    return trainer, datamodule


if __name__ == '__main__':
    # import our libraries
    from flash import Trainer
    from flash.text import TextClassifier, TextClassificationData
    EXPERIMENT_NAME = "text_classification"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print("Experiment_id: {}".format(experiment.experiment_id))
    mlflow.start_run(experiment_id=experiment.experiment_id)

    INPUT_FIELD = "review"
    TARGET_FIELD = "sentiment"
    TRAIN_FILE = "./data/imdb/train.csv"
    VAL_FILE = "./data/imdb/valid.csv"
    TEST_FILE = "./data/imdb/test.csv"
    PREDICT_FILE = "./data/imdb/predict.csv"
    BATCH_SIZE = 128

    # trainer(INPUT_FIELD, TARGET_FIELD, TRAIN_FILE, VAL_FILE, TEST_FILE, BATCH_SIZE)
    mlflow.end_run()

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="text_classification"):
    # 1. Init the finetuned task from URL

        model = TextClassifier.load_from_checkpoint("./text_classification_model.pt")

        # 2. Perform inference from list of sequences
        trainer, datamodule = trainer(INPUT_FIELD, TARGET_FIELD, TRAIN_FILE, VAL_FILE, TEST_FILE, BATCH_SIZE)

        predictions: list = trainer.predict(model, datamodule=datamodule, output="labels")

    # predictions = trainer.predict(model, datamodule=pred, output="labels")

