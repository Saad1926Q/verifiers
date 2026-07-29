import verifiers.v1 as vf
from verifiers.v1.harnesses.null import NullHarness


class FifteenPuzzleHarness(NullHarness):
    """Run each move from only the fixed instructions and current board."""

    async def resume(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
        messages: vf.Messages,
    ) -> vf.ProgramResult:
        return await self.launch(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data.model_copy(update={"prompt": messages}),
        )
