# scoring_project


Créer un répertoire à l'endroit souhaité
Sur terminal:
```
cd "[chemin d'accès du répertoire, par exemple: /Users/luciegabagnou/Documents/MOSEF/SCORING/projet]

```

Après s'être placé dans ce dossier via la commande cd, on va initialiser le dépôt git en local associé au projet sur Github

```
git clone https://github.com/luciegaba/scoring_project.git
git pull 
git checkout [prénom ==> correspond ta branche]
```
Après avoir fait ça, vous êtes placé dans votre branche (via checkout), et votre contenu est à jour par rapport au distant (via pull). Maintenant vous pouvez bosser sur votre branche. Quand vous avez fini de bosser, vous faites:
```
git add [nom du fichier travaillé]
git commit -m "[message de votre choix]
git push 
 
```
Le contenu est ensuite pushé (mis sur dépôt distant) sur ta branche. Il faut ensuite faire une demande de "merge" avec la branche "master". En fait ça permet de centraliser le contenu sur la branche principale. Si jamais a besoin de récupérer le contenu à jour, il le trouvera dessus. La merge request se fait en ligne. Il faudra ensuite réaligner son dépôt local en faisant git pull et git rebase master.



Créer un environnement virtuel 

```
conda create -n"[nom de l'environnemnt, exemple: projet]
conda activate [nom de l'environnement]
conda install python 
conda install pip
```

