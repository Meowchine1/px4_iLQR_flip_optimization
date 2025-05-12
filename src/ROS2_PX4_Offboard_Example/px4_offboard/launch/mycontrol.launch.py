import multiprocessing
import launch_ros
from launch import LaunchService
from launch import LaunchDescription
from launch import LaunchIntrospector
import time
import nest_asyncio
import os
import signal
from ament_index_python.packages import get_package_share_directory

nest_asyncio.apply()

num_parallel = 1

package_dir = get_package_share_directory('px4_offboard')

ld = [LaunchDescription([
        # ExecuteProcess(cmd=['bash', bash_script_path], output='screen'),
        launch_ros.actions.Node(
            package='px4_offboard',
            namespace='px4_offboard',
            executable='visualizer',
            name='visualizer',
            remappings=[('chatter', 'my_chatter'+str(i % num_parallel))]
        ),
        launch_ros.actions.Node(
            package='px4_offboard',
            namespace='px4_offboard',
            executable='processes',
            name='processes',
            prefix='gnome-terminal --',
            remappings=[('chatter', 'my_chatter'+str((i+1) % num_parallel))]
        ),
        launch_ros.actions.Node(
            package='px4_offboard',
            namespace='px4_offboard',
            executable='control',
            name='control',
            prefix='gnome-terminal --',
            remappings=[('chatter', 'my_chatter'+str((i+2) % num_parallel))]
        ),
        launch_ros.actions.Node(
            package='px4_offboard',
            namespace='px4_offboard',
            executable='velocity_control',
            name='velocity',
            remappings=[('chatter', 'my_chatter'+str((i+3) % num_parallel))]
        ),
        launch_ros.actions.Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', [os.path.join(package_dir, 'visualize.rviz')]],
            remappings=[('chatter', 'my_chatter'+str((i+4) % num_parallel))]
        )
    ]) for i in range(num_parallel)]

for item in ld:
    print(LaunchIntrospector().format_launch_description(item))

ls_pool = [LaunchService(noninteractive=True) for i in ld]
i = 0
for ls in ls_pool:
    ls.include_launch_description(ld[i])
    i += 1

proc_pool = [multiprocessing.Process(target=ls.run) for ls in ls_pool]

print('Starting')
for p in proc_pool:
    print('Stopping: {}'.format(p.pid))
    if not p.pid is None:
        os.kill(p.pid, signal.SIGINT)

for p in proc_pool:
    if not p.pid is None:
        p.join()

time.sleep(3)

ls_pool = [LaunchService(noninteractive=True) for i in ld]
i = 0
for ls in ls_pool:
    ls.include_launch_description(ld[i])
    i += 1

proc_pool = [multiprocessing.Process(target=ls.run) for ls in ls_pool]

print('Restarting...')

for p in proc_pool:
    p.start()

time.sleep(5)

print('Stopping')
for p in proc_pool:
    print('Stopping {}'.format(p.pid))
    os.kill(p.pid, signal.SIGINT)

for p in proc_pool:
    p.join()
