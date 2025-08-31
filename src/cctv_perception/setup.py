from setuptools import find_packages, setup

package_name = 'cctv_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mscrobotics2425laptop29',
    maintainer_email='mscrobotics2425laptop29@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'color_tracker = cctv_perception.color_tracker:main',
        	'tf_debug = cctv_perception.tf_debug:main',
        ],
    },
)
