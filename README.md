# uncertaintyinlatentspace

[![pipeline status](https://gitlab.michelin.com/Example: f123456/uncertaintyinlatentspace/badges/main/pipeline.svg)](https://gitlab.michelin.com/Example: f123456/uncertaintyinlatentspace/commits/main)
[![coverage report](https://gitlab.michelin.com/Example: f123456/uncertaintyinlatentspace/badges/main/coverage.svg)](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/coverage)
[![online documentation][badgeDoc]](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/docs/)
[![last tests report][badgeTest]](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/report_test.html)

SIM ready Python library.

## Code quality

[![Python 3.6 support badge](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/python3.6.svg)](https://gitlab.michelin.com/Example: f123456/uncertaintyinlatentspace/-/pipelines)
[![Python 3.9 support](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/python3.9.svg)](https://gitlab.michelin.com/Example: f123456/uncertaintyinlatentspace/-/pipelines)  
[![black badge](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/quality/black.svg)](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/quality/black.html)
[![isort badge](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/quality/isort.svg)](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/quality/isort.html)
[![pylint note](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/quality/pylint.svg)](http://Example: f123456.si-pages.michelin.com/uncertaintyinlatentspace/quality/pylint.html)

## Installation

You can install this package by running `pip install uncertaintyinlatentspace`.

For help setting up your python environment please refer to the
[Python Dynamics Documentation][pydnx-documentation].

## Setup a development environment

Inside a dedicated virtualenv, run the following:
```bash
pip install pydnx-packaging
pip install -e .[dev]
```

## Best practices

To ensure that your code is up to python standards, please apply `black` and `isort` **before each commit**:
```bash
black .
isort .
```
These tools will do the job **automatically** for you.

To go further and dive into code quality you can use:
```bash
pylint uncertaintyinlatentspace
pydocstyle
```

## CI-CD

This project uses the [Central CI-CD](https://gitlab.michelin.com/dord/infra/central-ci-cd).

Check it out for documentation and support.

## README Badges

1. If the project is owned by a group you have to replace `Example: f123456` by the group ID in this `README.md` file.

2. If the badges do not display correctly, check that their links use the correct url
   (project url and branch name).

[badgeDoc]: https://img.shields.io/badge/-online_documentation-grey.svg?logoColor=white&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAASCAQAAAD8x0bcAAAAZUlEQVR4AbXBAQbCYBQA4Bdm59xFRlC7QWFH2KWG8g8IRoEv8Age0L4vjuHkrGnGqJmkS1S8pEdUNOkZFVdpiprRZjPG0fQGixWsFoM+funMdintZl0kN5V7JB+VdyQ1kdTEX30BH5bT1ofC5y4AAAAASUVORK5CYII=
[badgeTest]: https://img.shields.io/badge/-last_tests_report-grey.svg?logoColor=white&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAkCAYAAADhAJiYAAABMElEQVR42u1WuwrCQBAMBKxtrBUJdv6DpSCCjfovSWFQYql+g7X4MRI7bX30WimJe3DCErz1LuZEYQeGFJmdm+RuN3EcBoPBYPwx0jR1gU1gHxgAwxdsIX1LofGlh/By84YZAs/A5A1DVBNq6IXn0DRMpGGcN9CTkW6YABcCdnCdAttyS7KsodqaQiPYAa9lJlTwLowHvKOCGbBU8LnsAm/SX6zlUeIZCrOy2CwTtM5cKYRXukHCqsIsNKHCo4zW2VLJr/LcHAhNYkLi4Q9yrQv1hi5SdLQdCO6ddALFyKxiccvqqItjassWKNDY4qFe406mhA3U9qI1ewUHKYmuQmHotpdFfmYwimHWIQaezmBsy+G6NxqMyHj0hU/H6H8/rhZ+P0Tt4KPfDwaDwWD8CB70tUqqukm2zQAAAABJRU5ErkJggg==
[pydnx-documentation]: http://pydnx.si-pages.michelin.com/documentation/
