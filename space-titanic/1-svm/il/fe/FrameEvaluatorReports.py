from .FrameEvaluatorReport import FrameEvaluatorReport


class FrameEvaluatorReports:
    def __init__(self):
        self.reports: FrameEvaluatorReport = []

    def add(self, report: FrameEvaluatorReport):
        self.reports.append(report)

    def print(self):
        sorted_reports = sorted(
            self.reports, key=lambda report: report.scores.mean(), reverse=True)

        for report in sorted_reports:
            report.print()
