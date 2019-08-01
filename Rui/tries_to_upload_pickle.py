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


def send_to_drive():
	try:
        g_login = GoogleAuth()
        g_login.LocalWebserverAuth()
		with open(r"D:\Physionet Challenge\GitHub\pnet2019\Rui\grid_search_object.p","r") as file:
            file_drive = drive.CreateFile({'grid_search_object.p':os.path.basename(file.name) })
            file_drive.SetContentString(file.read())
            file_drive.Upload()
		print("Drive Worked!")
	except:
	    print('Drive did not work...')


time_sleep = 3600 * 2
while "grid_search_object.p" not in os.listdir(r"D:\Physionet Challenge\GitHub\pnet2019\Rui"):
    time.sleep(time_sleep)

# git_push()
send_to_drive()

print("It's all over!")
