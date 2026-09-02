from reachy_mini_conversation_app.profile_store import read_packaged_default_profile


CLOUD_SPACE_TOOLS = (
    "pollen_robotics_reachy_mini_search_tool__search_web",
    "pollen_robotics_reachy_mini_weather_tool__get_weather",
    "pollen_robotics_reachy_mini_time_tool__get_time",
)


def test_default_profile_keeps_local_reachy_tools_without_cloud_spaces() -> None:
    """Local conversation must not enable hosted weather/search/time by default."""
    tools = read_packaged_default_profile().default_tools
    assert "dance" in tools
    assert "move_head" in tools
    assert "head_tracking" in tools
    assert "camera" in tools
    assert "go_to_sleep" in tools
    assert "home_assistant" in tools
    assert "monitor_bus" in tools
    assert "apex" in tools
    assert "ask_hermes" in tools
    assert not set(CLOUD_SPACE_TOOLS) & set(tools)


def test_default_profile_sends_tank_trends_to_ask_hermes() -> None:
    """Trend/thread questions must go to Hermes, not the live Apex snapshot tools."""
    instructions = read_packaged_default_profile().instructions
    assert "ask_hermes" in instructions
    assert "tank trends" in instructions
    assert "do not call apex or reef_status first" in instructions.lower()
    assert "still on" in instructions.lower()
    assert "do not use ask_hermes for live reef status" in instructions.lower()
    assert "source=cache" in instructions
    assert "stale=true" in instructions
    assert "handle that new request" in instructions.lower()


def test_default_profile_sends_live_tank_status_to_apex() -> None:
    """Current tank status must use the local Apex snapshot, not Hermes."""
    instructions = read_packaged_default_profile().instructions
    assert "reef tank status" in instructions.lower()
    assert "use **apex** immediately" in instructions.lower()
    assert "call **home_assistant** or **monitor_bus** immediately" in instructions.lower()
    assert "call **monitor_bus** immediately" in instructions.lower()
    assert "button.screen_down" in instructions
    assert "do not call play_emotion for that success" in instructions.lower()
    assert "done, lamp three is on" in instructions.lower()
