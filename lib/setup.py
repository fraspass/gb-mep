from setuptools import setup

setup(
	name="recsys",
	version="0.1",
	packages=[
		"recsys",
	],
	install_requires=[
		"numpy",
		"scipy",
		"numba",
	],
)
