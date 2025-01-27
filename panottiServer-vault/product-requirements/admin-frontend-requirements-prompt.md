I have a python-based FastAPI server running in the root directory of the repository.  I've added a Next.js application running inside the repository in a folder called '/admin-frontend'.  This application will be an admin frontend for the FastAPI server that allows the user to change variables, view the logs and restart the server.

This is a basic Next.js application that has a basic layout and style which we want to preserve.

# New Features

I would like to add the following features to the next.js app to manage the app:
## Basic admin login
- default basic password to login.  I want to have a basic password that's stored as a hash value in a file in the root of the application.  this will default to the word 'Pa55w0rd'. 
- The user should be able to change the password in a new 'Admin' menu, which will update the hash in the file.
- The user should be prompted to change the default password when they first login, so you might want to make the password changing functionality a component that can run inside a model.
- The user should not be able to access any page on the website unless they supply the correct password, which will then be hashed and compared against the one saved in the file.

## Settings Page

I'd like you to add a component to the home page called 'Settings'.  This will be an accordion component that will allow the user to read and write settings in the app.

To populate the accordion, I want you to look at the root repository directory one level up from the next.js root '../' where the FastAPI server code lives.  The plugins directory can be found under app/plugins



