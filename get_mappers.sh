# script that fetches Semantify mappers

# go to semantify directory
cd semantify

# download the models & extract the data
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1KZRSvSny-BD7fbXvogkSwN8tEvnzlYOa" -o semantify_mappers.log && unzip 'uc?export=download&id=1KZRSvSny-BD7fbXvogkSwN8tEvnzlYOa' && rm 'uc?export=download&id=1KZRSvSny-BD7fbXvogkSwN8tEvnzlYOa' semantify_mappers.log
