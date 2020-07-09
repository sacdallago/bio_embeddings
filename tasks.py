from invoke import task
from shutil import which
import os,sys

@task
def clean(c, bytecode=False, extra=''):
    patterns = ['build']
    if bytecode:
        patterns.append('**/*.pyc')
    if extra:
        patterns.append(extra)
    for pattern in patterns:
        c.run("rm -rf {}".format(pattern))

@task
def build(c, update_requirements=True):
    poetry = which("poetry") is not None
    python_command = "poetry run python" if poetry else "python"

    if update_requirements:
        if poetry:
            c.run("poetry export -f requirements.txt > requirements.txt")
        else: 
            c.run("pip freeze > requirements.txt")

    # Create dist files
    if not poetry:
        c.run(f"{python_command} setup.py sdist")
        c.run(f"{python_command} setup.py bdist_wheel")
    else: 
        c.run("poetry build")
