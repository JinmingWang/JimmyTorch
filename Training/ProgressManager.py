from rich.console import Console
from rich.live import Live
from rich.table import Table
from threading import Thread
import time
from typing import List

class ProgressManager:
    def __init__(self, items_per_epoch: int, epochs: int, show_recent: int, refresh_interval: int = 1, custom_fields: List[str] = None):
        """
        :param dataloader: the data loader
        :param epochs: the total number of epochs
        :param show_recent: how many recent epochs to display
        :param description: the texts showing in front of progress bar
        """
        self.epochs = epochs
        self.display_recent = show_recent
        self.steps_per_epoch = items_per_epoch
        self.refresh_interval = refresh_interval
        self.custom_fields = [] if custom_fields is None else custom_fields

        # Initialize tracking
        self.overall_progress = 0
        self.start_time = time.time()  # Start tracking time
        self.console = Console(width=120)
        self.live = None

        self.current_epoch = 1
        self.current_step = 1

        # Initialize progress data for recent epochs
        self.memory = [
            ({"epoch": epoch, "completed": 0, "t_start": 0.0, "t_end": 0.0} | dict(zip(self.custom_fields, [0]*len(custom_fields))))
            for epoch in range(1, epochs + 1)
        ]

    def __enter__(self):
        """Enter the context: start displaying the progress."""
        self.console.print("[bold green]Starting Training...[/bold green]")
        self.live = Live(self.render_progress_table(1), refresh_per_second=1, console=self.console)
        self.live.__enter__()
        self.live_thread = Thread(target=self.live_update)
        self.thread_stop = False
        self.start_time = time.time()
        self.live_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context: stop the progress and display the completion message."""
        self.console.print("[bold green]Training Completed![/bold green]")
        self.thread_stop = True
        self.live.__exit__(exc_type, exc_val, exc_tb)

    def update(self, current_epoch: int, current_step: int, **custom_values):
        """Update the progress of the current epoch and overall progress."""
        # Update overall progress
        self.overall_progress += 1

        # Update the specific epoch progress
        self.memory[current_epoch]["completed"] = current_step + 1
        for k in self.custom_fields:
            self.memory[current_epoch][k] = custom_values[k]

        self.current_epoch = current_epoch + 1
        self.current_step = current_step + 1

        if self.memory[current_epoch]["t_start"] == 0:
            # we are starting a new epoch, record the start time
            self.memory[current_epoch]["t_start"] = time.time()
            if current_epoch >= 1:
                # update the end time of the previous epoch
                self.memory[current_epoch - 1]["t_end"] = time.time()


    def format_time(self, seconds: float) -> str:
        """Format time in seconds into hh:mm:ss."""
        hrs, rem = divmod(seconds, 3600)
        mins, secs = divmod(rem, 60)
        return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

    def render_progress_table(self, current_epoch: int) -> Table:
        """Create a table to display overall and recent epoch progress."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Desc", width=10)
        table.add_column("Percent", width=7)
        table.add_column("Progress", width=10)
        table.add_column("Elapsed", width=8)
        table.add_column("Remaining", width=9)
        for k in self.custom_fields:
            table.add_column(k, width=12)

        # Calculate time details
        elapsed_time_total = time.time() - self.start_time
        total_steps = self.epochs * self.steps_per_epoch
        remaining_time_total = (total_steps - self.overall_progress) * (
                    elapsed_time_total / self.overall_progress) if self.overall_progress > 0 else 0

        # Overall progress row
        overall_percentage = self.overall_progress / total_steps * 100
        table.add_row(
            "[#00aaff]Overall[#00aaff]",
            f"[#00aaff]{overall_percentage:.2f}%[/#00aaff]",
            f"[#00aaff]{self.overall_progress}/{total_steps}[/#00aaff]",
            f"[#00aaff]{self.format_time(elapsed_time_total)}[/#00aaff]",
            f"[#00aaff]{self.format_time(remaining_time_total)}[/#00aaff]",
            "",
        )

        # Display the recent epochs
        for i in range(max(0, current_epoch - self.display_recent), current_epoch):
            epoch_data = self.memory[i]
            epoch_percentage = epoch_data["completed"] / self.steps_per_epoch * 100
            if epoch_data["t_end"] == 0:
                # epoch is not completed yet
                elapsed_time_epoch = time.time() - epoch_data["t_start"]
                remaining_time_epoch = (self.steps_per_epoch - epoch_data["completed"]) * (
                            elapsed_time_epoch / epoch_data["completed"]) if epoch_data["completed"] > 0 else 0
                complete_color = "green" if remaining_time_epoch == 0 else "#ffff00"
            else:
                # epoch is completed
                elapsed_time_epoch = epoch_data["t_end"] - epoch_data["t_start"]
                remaining_time_epoch = 0
                complete_color = "green"
            table.add_row(
                f"[{complete_color}]Epoch {epoch_data['epoch']}[/{complete_color}]",
                f"[{complete_color}]{epoch_percentage:.2f}%[/{complete_color}]",
                f"[{complete_color}]{epoch_data['completed']}/{self.steps_per_epoch}[/{complete_color}]",
                f"[{complete_color}]{self.format_time(elapsed_time_epoch)}[/{complete_color}]",
                f"[{complete_color}]{self.format_time(remaining_time_epoch)}[/{complete_color}]",
                *[f"{epoch_data[k]:.5e}" for k in self.custom_fields]
            )

        return table

    def live_update(self):
        while not self.thread_stop:
            self.live.update(self.render_progress_table(self.current_epoch))
            time.sleep(self.refresh_interval)