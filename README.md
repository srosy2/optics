# optics
<br>hakaton </br>
git clone "названия репозитория", чтобы получить данные на компьютер  
Если хотите создать что-то новое, создайте новую ветку: git pull, git checkout -b "название", git add "название файла", git commit -m "сообщение", git push  
Запускать через check_os.py  
Менять линзу и данные среды в файле ray_optics_criteria_ITMO.py, в функции test_opt_model.  

<br>Paralell optuna</br>

docker pull postgres
docker run -d --name postgres-container -e POSTGRES_PASSWORD=123 -p 5432:5432 postgres
docker exec -it postgres-container bash
psql -U postgres
CREATE DATABASE optics;
\c optics
\q
exit

