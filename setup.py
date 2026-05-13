from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'diffusion_policy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*'))),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='michaelbezick',
    maintainer_email='michaelbezick@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gcrl_manipulation_node = diffusion_policy.gcrl_manipulation_node:main',
            'one_robot_gcrl_manipulation_node = diffusion_policy.one_robot_gcrl_manipulation_node:main',
            'dp_inference_node = diffusion_policy.dp_inference_node:main',
        ],
    },
)
