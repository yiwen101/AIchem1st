from app.tool.base import BaseTool


_REPORT_STUCK_DESCRIPTION = """Report that the AI is stuck in a loop or cannot make progress on the task.
Use this tool when you detect you are repeating the same actions without progress,
facing an insurmountable obstacle, or otherwise cannot continue with the task.
This will terminate the interaction with a 'stuck' status."""


class ReportStuck(BaseTool):
    name: str = "report_stuck"
    description: str = _REPORT_STUCK_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "A clear explanation of why the AI is stuck and cannot proceed with the task.",
            }
        },
        "required": ["reason"],
    }

    async def execute(self, reason: str) -> str:
        """Report that the AI is stuck and terminate the interaction"""
        return f"The interaction has been terminated due to being stuck: {reason}"
