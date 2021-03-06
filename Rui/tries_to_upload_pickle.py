from git import Repo
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from time import sleep

def git_push():
    path_to_repo = r"D:\Physionet Challenge\GitHub\pnet2019\.git"
    commit_message = "Added pickle of the model of GridSearch."
    try:
        repo = Repo(path_to_repo)
        repo.git.add(A=True)
        repo.index.commit(commit_message)
        origin = repo.remote(name='origin')
        origin.push()
        print("Git Worked!")
    except:
        print('Git did not work...') 


def git_push_v2():
    try:
        path_to_repo = r"D:\Physionet Challenge\GitHub\pnet2019\.git"
        commit_message = "Put results in a folder and fixed the type of a file that had no type"	
        repo = Repo(path_to_repo)
        git_ = repo.git
        git_.add(A=True)
        git_.commit(m=commit_message)
        git_.push()
        print("Git Worked!")
    except:
        print("This version of git didn't work... (This one could only work if the previous didn't.)")


def send_to_drive():
    try:
        g_login = GoogleAuth()
        g_login.LocalWebserverAuth()
        drive = GoogleDrive(g_login)
        with open(r"D:\Physionet Challenge\GitHub\pnet2019\Rui\grid_search_object.p","r") as file:
            file_drive = drive.CreateFile({'grid_search_object.p':os.path.basename(file.name) })
            file_drive.SetContentFile(file.read())
            file_drive.Upload()
        print("Drive Worked!")
    except:
        print('Drive did not work...')


time_sleep = 3600 * 2
while "grid_search_object.p" not in os.listdir(r"D:\Physionet Challenge\GitHub\pnet2019\Rui"):
    print("Let me sleep...")
    sleep(time_sleep)

git_push()
git_push_v2
send_to_drive()

print("It's all over!")
