@ECHO OFF

if "%1" == "lint" goto lint
if "%1" == "fmt" goto fmt
if "%1" == "mypy" goto mypy
if "%1" == "install" goto install
goto end

:lint
flake8 ignite tests examples --config setup.cfg
ufmt diff .
goto end

:fmt
ufmt format .
goto end

:mypy
mypy --config-file mypy.ini
goto end

:install
pip install --upgrade flake8 "black==21.12b0" "usort==1.0.5" "ufmt==2.0.1" "mypy"
goto end

:end
popd
