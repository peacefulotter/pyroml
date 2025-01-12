rm -rf ./dist
rm -rf pyroml.egg-info
python3 -m build
twine check dist/*
twine upload -r testpypi dist/*
twine upload dist/*
