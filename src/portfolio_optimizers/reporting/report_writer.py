import pandas as pd
from pathlib import Path

class OptimizerReportWriter:
    def __init__(self, base_dir: Path, output_file: str | None = None):
        self.report_dir = Path(base_dir)
        self.output_file = output_file or f"optimizer_report_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write_report(self, results: pd.DataFrame) -> str:
        """Write the optimization results to an Excel report."""
        period_summary = (
            results[["Max_Time_Horizon", "Risk_Aversion", "Period", "Period_Return", "Active_Positions", "Turnover_Pct", "Objective_Value"]]
            .drop_duplicates()
            .sort_values(["Max_Time_Horizon", "Risk_Aversion", "Period"])
        )

        weights = (
            results
            .pivot_table(
                index=["Max_Time_Horizon", "Risk_Aversion", "Period"],
                columns="Security",
                values="Weight",
                aggfunc="first",
            )
            .reset_index()
        )

        output_path = self.report_dir / self.output_file
        with pd.ExcelWriter(output_path) as writer:
            results.to_excel(writer, sheet_name="Results", index=False)
            period_summary.to_excel(writer, sheet_name="Period Summary", index=False)
            weights.to_excel(writer, sheet_name="Weights", index=False)

        print(f"Optimizer report saved to {output_path}")
        return str(output_path)