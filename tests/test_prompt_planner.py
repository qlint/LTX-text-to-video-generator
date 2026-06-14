from app.core.config import Settings
from app.services.prompt_planner import PromptPlanner


def test_portrait_prompt_uses_portrait_dimensions() -> None:
    planner = PromptPlanner(Settings())
    plan = planner.plan_batch(["A vertical TikTok-style cooking shot."], scene_seed=123)[0]

    assert plan.orientation == "portrait"
    assert plan.target_width == 1080
    assert plan.target_height == 1920


def test_long_prompt_is_segmented() -> None:
    settings = Settings(max_prompt_words_per_segment=10, minimum_seconds_per_video=1)
    planner = PromptPlanner(settings)
    prompt = " ".join(f"word{i}" for i in range(45))

    plan = planner.plan_batch([prompt], scene_seed=123)[0]

    assert len(plan.segments) > 1
    assert all(segment.num_frames % 8 == 1 for segment in plan.segments)

