import os
from glob import glob
from setuptools import setup

package_name = 'px4_offboard'

#create log/position directory

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/ament_index/resource_index/packages',
            ['resource/' + 'visualize.rviz']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('resource/*rviz'))
        # (os.path.join('share', package_name), ['scripts/TerminatorScript.sh'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Braden',
    maintainer_email='braden@arkelectron.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'offboard_control = px4_offboard.offboard_control:main',
                'offboard_control_example = px4_offboard.offboard_control_example:main',
                'visualizer = px4_offboard.visualizer:main',
                'processes = px4_offboard.processes:main',
                'velocity_control = px4_offboard.velocity_control:main',
                'custom_control = px4_offboard.custom_control:main',
                #'base_control_test = px4_offboard.base_control_test:main'
               # 'px4_interaction = px4_offboard.px4_interaction:main',
               # 'dynamics_drone_control = px4_offboard.dynamics_drone_control:main'
                 
        ],
    },
)
