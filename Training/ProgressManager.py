import threading

from rich.console import Console
from rich.live import Live
from rich.table import Table, Column
from threading import Thread
import time
from typing import List

class ProgressManager:
    def __init__(self, items_per_epoch: int, epochs: int, show_recent: int, refresh_interval: int = 1,
                 custom_fields: List[str] = None):
        """
        :param dataloader: the data loader
        :param epochs: the total number of epochs
        :param show_recent: how many recent epochs to display
        :param refresh_interval: how often to refresh the display (in seconds)
        :param description: the texts showing in front of progress bar
        """
        self.epochs = epochs
        self.steps_per_epoch = items_per_epoch
        self.total_steps = epochs * items_per_epoch
        self.display_recent = show_recent
        self.refresh_interval = refresh_interval
        self.custom_fields = [] if custom_fields is None else custom_fields

        self.total_width = (9 + 9 + 15 + 10 + 10) + 12 * len(self.custom_fields)

        # Initialize tracking
        self.overall_progress = 0
        self.start_time = time.time()  # Start tracking time
        self.console = Console(width=self.total_width + 6 + len(self.custom_fields))
        self.live = None

        self.current_epoch = 1
        self.current_step = 1

        # Initialize progress data for recent epochs
        self.memory = [
            ({"epoch": epoch, "completed": 0, "t_start": 0.0, "t_end": 0.0} | dict(zip(self.custom_fields, [0]*len(custom_fields))))
            for epoch in range(1, epochs + 1)
        ]

        self.overall_row_forms = [
            "[#00aaff]Overall[#00aaff]",    # Overall row Description
            "[#00aaff]{}%[/#00aaff]",       # Overall row Percentage
            "[#00aaff]{}/{}[/#00aaff]",     # Overall row Progress
            "[#00aaff]{}[/#00aaff]",        # Overall row Elapsed time
            "[#00aaff]{}[/#00aaff]",        # Overall row Remaining time
            ]

    def update(self, current_epoch: int, current_step: int, **custom_values):
        if not hasattr(self, "live_thread"):
            self.console.print("[bold green]Starting Training...[/bold green]")
            self.live = Live(self.render_progress_table(1), refresh_per_second=1, console=self.console)
            self.start_time = time.time()
            self.live_thread = Thread(target=self.live_update)
            self.live_thread.start()

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
        hrs, rem = divmod(int(seconds), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hrs:02}:{mins:02}:{secs:02}"

    def render_progress_table(self, current_epoch: int) -> Table:
        """Create a table to display overall and recent epoch progress."""
        table = Table(show_header=True, header_style="bold magenta", min_width=self.total_width)
        table.add_column("Epoch", width=9)
        table.add_column("Percent", width=9)
        table.add_column("Progress", width=15)
        table.add_column("Elapsed", width=10)
        table.add_column("Remain", width=10)
        for k in self.custom_fields:
            table.add_column(k, width=12)

        # Calculate time details
        elapsed_time_total = time.time() - self.start_time
        remaining_time_total = (self.total_steps - self.overall_progress) * (
                elapsed_time_total / self.overall_progress) if self.overall_progress > 0 else 0

        # Overall progress row
        overall_percentage = int(self.overall_progress / self.total_steps * 100)
        table.add_row(self.overall_row_forms[0],
                      self.overall_row_forms[1].format(overall_percentage),
                      self.overall_row_forms[2].format(self.overall_progress, self.total_steps),
                      self.overall_row_forms[3].format(self.format_time(elapsed_time_total)),
                      self.overall_row_forms[4].format(self.format_time(remaining_time_total)),
                      "")

        # Display the recent epochs
        for i in range(max(0, current_epoch - self.display_recent), current_epoch):
            epoch_data = self.memory[i]
            epoch_percentage = int(epoch_data["completed"] / self.steps_per_epoch * 100)
            if epoch_data["t_end"] == 0:
                # epoch is not completed yet
                elapsed_time_epoch = time.time() - epoch_data["t_start"]
                remaining_time_epoch = (self.steps_per_epoch - epoch_data["completed"]) * (
                            elapsed_time_epoch / epoch_data["completed"]) if epoch_data["completed"] > 0 else 0
                color = "green" if remaining_time_epoch == 0 else "#ffff00"
            else:
                # epoch is completed
                elapsed_time_epoch = epoch_data["t_end"] - epoch_data["t_start"]
                remaining_time_epoch = 0
                color = "green"
            table.add_row(
                f"[{color}]{epoch_data['epoch']}[/{color}]",
                f"[{color}]{epoch_percentage}%[/{color}]",
                f"[{color}]{epoch_data['completed']}/{self.steps_per_epoch}[/{color}]",
                f"[{color}]{self.format_time(elapsed_time_epoch)}[/{color}]",
                f"[{color}]{self.format_time(remaining_time_epoch)}[/{color}]",
                *[f"{epoch_data[k]:.3e}" for k in self.custom_fields]
            )

        return table

    def live_update(self):
        with self.live:
            while self.overall_progress != self.total_steps:
                # Update the live display
                self.live.update(self.render_progress_table(self.current_epoch))
                time.sleep(self.refresh_interval)


    def close(self):
        """Close the live display."""
        if hasattr(self, "live_thread"):
            self.live.stop()
            self.live_thread.join()
            del self.live_thread
            del self.live
            self.console.print("[bold green]Training Completed![/bold green]")