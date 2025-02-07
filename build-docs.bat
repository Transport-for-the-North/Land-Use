call activate normits_docs
rmdir /s /q build
sphinx-build -M html docs build
call activate normits_lu
