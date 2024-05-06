from cliport.environments.environment import Environment
import pybullet as p

import os
import tempfile
import time
import cv2
import imageio

import gym
import numpy as np
from cliport.tasks import cameras
from cliport.utils import pybullet_utils
from cliport.utils import utils

class EnvironmentTopDown(Environment):
    """
    Environment wiht top-down view cameras.
    """

    def __init__(self,
                 assets_root,
                 task=None,
                 disp=False,
                 shared_memory=False,
                 hz=240,
                 record_cfg=None):
        """Creates OpenAI Gym-style environment with PyBullet.

        Args:
          assets_root: root directory of assets.
          task: the task to use. If None, the user must call set_task for the
            environment to work properly.
          disp: show environment with PyBullet's built-in display viewer.
          shared_memory: run with shared memory.
          hz: PyBullet physics simulation step speed. Set to 480 for deformables.

        Raises:
          RuntimeError: if pybullet cannot load fileIOPlugin.
        """
        self.pix_size = 0.003125
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.agent_cams = cameras.Oracle310.CONFIG
        self.record_cfg = record_cfg
        self.save_video = False
        self.step_counter = 0

        self.assets_root = assets_root

        color_tuple = [
            gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
            for config in self.agent_cams
        ]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        })
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5, 0.], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })

        # Start PyBullet.
        disp_option = p.DIRECT
        if disp:
            disp_option = p.GUI
            if shared_memory:
                disp_option = p.SHARED_MEMORY
        client = p.connect(disp_option)
        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / hz)

        # If using --disp, move default camera closer to the scene.
        if disp:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target)

        if task:
            self.set_task(task)
        
