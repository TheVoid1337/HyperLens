

def generate_stage_configurations(hidden_channels: int, number_of_layers: int, double_steps: int, plateau_size: int,
                                  max_channels: int = None) -> tuple[list[int], list[int]]:
    channel_list = []
    depth_list = []

    for stage_index in range(number_of_layers):
        if stage_index <= double_steps:
            exponent = stage_index
        else:
            exponent = double_steps + (stage_index - double_steps) // plateau_size

        current_channels = hidden_channels * (2 ** exponent)

        if max_channels is not None:
            current_channels = min(current_channels, max_channels)

        if stage_index == 0:
            stage_depth = 0
        elif stage_index < double_steps:
            stage_depth = 4
        elif stage_index == double_steps:
            stage_depth = 8
        else:
            stage_depth = 2

        channel_list.append(current_channels)
        depth_list.append(stage_depth)

    return channel_list, depth_list


