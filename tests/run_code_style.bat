@ECHO OFF

if "%1" == "lint" goto lint
if "%1" == "fmt" goto fmt
if "%1" == "mypy" goto mypy
if "%1" == "install" goto install
goto end

:lint
flake8 ignite tests examples --config setup.cfg
isort . --check --settings setup.cfg
black . --check --config pyproject.toml
goto end

:fmt
isort . --settings setup.cfg
black . --config pyproject.toml
goto end

:mypy
mypy --config-file mypy.ini
goto end

:install
pip install flake8 "black==19.10b0" "isort==5.7.0" mypy
goto end

:end
popd
