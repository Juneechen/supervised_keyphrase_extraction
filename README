### Collboration Guide

1.  `git clone https://github.com/Juneechen/bi-LSTM.git`

2.  Create a branch:

        cd bi-LSTM
        git checkout -b <branch_name>

3.  Make Changes:

        git add .
        git commit -m "commit message"

4.  Push Changes to Remote:

        git push origin <branch_name>

5.  Create a Pull Request (PR):  
    Go to the repository on GitHub, you should see an option to create a pull request directly from your branch. Click on it and choose `main` to merge into.

6.  Update Local Repository:
    After the pull request is merged, it's a good idea to update your local repository with the latest changes from the remote:

        git checkout main
        git pull origin main

7.  After you have checked out the main branch and pulled the latest changes, you can switch back to your working branch using the following commands:

            git checkout <your_branch_name>

    If you want to bring the latest changes from main into your branch, you'll need to merge or rebase your branch onto main. Here are two common approaches:

    Merging

            # Make sure you are on your working branch
            git checkout <your_branch_name>

            # Merge changes from main into your branch
            git merge main

    Rebasing

        # Make sure you are on your working branch
        git checkout <your_branch_name>

        # Rebase your branch onto main
        git rebase main

    Merging creates a new commit to integrate changes, while rebasing replays your branch's changes on top of main, resulting in a linear commit history.
