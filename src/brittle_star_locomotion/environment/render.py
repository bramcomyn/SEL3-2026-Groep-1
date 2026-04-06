import jax
import numpy as np

import mediapy as media
from tqdm import tqdm
from pathlib import Path

from brittle_star_locomotion.logger.logger import Logger
from brittle_star_locomotion.config.configuration import Configuration



class EnvironmentRenderer:
    """
    Renderer for the brittle star locomotion environment.
    """
    def __init__(self, environment):
        self.environment = environment
        self.logger = Logger() # TODO: get logger
        self.config = Configuration().configuration

    def render_video(self, trajectory, output_path: str = "out/test-video.mp4"):
        """Processes trajectory for ALL environments and stacks them vertically."""
        self.logger.info(f"Rendering all environments to {output_path}")
        frames = []

        # dimensions: (n_envs, steps, ...)
        first_leaf = jax.tree_util.tree_leaves(trajectory)[0]
        total_steps = first_leaf.shape[1]
        num_steps = first_leaf.shape[0]
        render_indices = range(0, total_steps, self.config.env.render_every)

        for i in tqdm(render_indices, desc="Generating Video Frames"):
            env_frames_for_this_step = []
            
            for e in range(num_steps):
                # extract state for step 'e' and time 'i'
                step_state = jax.tree_util.tree_map(lambda x: x[e, i], trajectory)
                brittle_star_environment = self.environment.brittle_star_environment
                raw_frames = brittle_star_environment.render(step_state) 
                
                if raw_frames is not None:
                    processed_list = [np.asarray(f) for f in raw_frames]
                    combined_camera_view = self.__post_render(processed_list)
                    env_frames_for_this_step.append(combined_camera_view)

            frames.extend(env_frames_for_this_step)

        if frames:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            media.write_video(str(output_file), np.array(frames), fps=20)
            self.logger.info(f"Saved multi-env video ({len(frames)} frames) to {output_path}")

    def show_video(self, video_path: str):
        """display the video in a notebook environment."""
        if Path(video_path).exists():
            media.show_video(media.read_video(video_path))

    def __post_render(self, render_output: list[np.ndarray]) -> np.ndarray | None:
        """converts list of camera arrays into a single stitched array."""
        if render_output is None or len(render_output) == 0:
            return None

        num_cameras = len(self.config.env.camera_ids) # TODO: get from config instead of env? or pass as argument

        # If we have multiple cameras, stitch them side-by-side (axis=1)
        if num_cameras > 1:
            processed_frame = np.concatenate(render_output, axis=1)
        else:
            processed_frame = render_output[0]

        return processed_frame[:, :, ::-1]  # convert from RGB to BGR for mediapy