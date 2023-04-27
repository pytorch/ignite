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
pip install --upgrade flake8 "black==23.3.0" "usort==1.0.6" "ufmt==2.1.0" "mypy"
goto end

:end
popd
