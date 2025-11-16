import os
from typing import *
from rich.tree import Tree
from rich.console import Console

RecordStatus = Literal["Incomplete", "Trained", "Tested"]
RunRecord = Tuple[RecordStatus, str]
ModelRecord = dict[str, RunRecord]
DatasetRecord = dict[str, ModelRecord]


class RunRecordManager:
    def __init__(self, root: str = "./Runs"):
        self.records: dict[str, DatasetRecord] = dict()
        for dataset_name in os.listdir(root):
            dataset_path = os.path.join(root, dataset_name)
            self.records[dataset_name]: DatasetRecord = self.__processDataset(dataset_path)


    def __processRecord(self, path: str) -> RunRecord:
        files_inside = os.listdir(path)
        if "best.pth" not in files_inside:
            status = "Incomplete"
        elif any([name.endswith(".csv") for name in files_inside]):
            status = "Tested"
        else:
            status = "Trained"

        if status == "Incomplete":
            comments = "N/A"
        else:
            with open(os.path.join(path, "comments.txt"), "r") as f:
                comments = f.readline().strip()

        return status, comments



    def __processModel(self, path: str) -> ModelRecord:
        records: ModelRecord = dict()
        for run_name in os.listdir(path):
            run_path = os.path.join(path, run_name)
            records[run_name]: RunRecord = self.__processRecord(run_path)

        return records


    def __processDataset(self, path: str) -> DatasetRecord:
        records: DatasetRecord = dict()
        for model_name in os.listdir(path):
            dataset_path = os.path.join(path, model_name)
            records[model_name]: ModelRecord = self.__processModel(dataset_path)

        return records


    def printRecords(self, datasets: Optional[List[str]] = None, models: Optional[List[str]] = None):
        console = Console(width=120)
        root_tree = Tree("[bold blue]Runs[/bold blue]")
        if datasets is None:
            datasets = self.records.keys()

        for dataset_name in datasets:
            dataset_rec = self.records.get(dataset_name, {})
            dataset_tree = root_tree.add(f"[bold #00FFFF]{dataset_name}[/bold #00FFFF]")

            if models is None:
                models = dataset_rec.keys()

            for model_name in models:
                model_rec = dataset_rec.get(model_name, {})
                model_tree = dataset_tree.add(f"[magenta]{model_name}[/magenta]")
                for run_name, run_rec in model_rec.items():
                    status_color = {
                        "Incomplete": "#FF0000",
                        "Trained": "#FFFF00",
                        "Tested": "#00FF00",
                    }.get(run_rec[0], "white")
                    model_tree.add(
                        f"{run_name}: "
                        f"[{status_color}]{run_rec[0]}[/], "
                        f"[italic]{run_rec[1]}[/italic]"
                    )

        console.print(root_tree)


if __name__ == "__main__":
    rrm = RunRecordManager()
    rrm.printRecords()
