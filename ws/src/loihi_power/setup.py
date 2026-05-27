from setuptools import find_packages, setup

package_name = 'loihi_power'

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
    maintainer='meteorlake01-gdc',
    maintainer_email='leobardo.e.campos.macias@intel.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'loihi_power_node = loihi_power.loihi_power_node:main',
        ],
    },
)
