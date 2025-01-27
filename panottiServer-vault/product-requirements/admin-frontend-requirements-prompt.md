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

I want you to:
1. loop over all the plugin directories in the ../app/plugins in the FastAPI server
2. parse both the plugin.yaml files AND parse their corresponding plugin.yaml.example files
3. fore each plugin you find, which can be named by looking at the name in the plugin.yaml file (example - name: "audio_transcription_local"), add an accordion section for each plugin, with the title being the plugin name, remove the underscores in the names and sentence-case the titles.
4. Inside each accordion component, I want you to create a form unique to that plugin
5. inside the form, for each component, I want you to create a text field for every config value you find in the "config" section of the yaml.  At the top of the form I want you to add an "enabled" checkbox representing the value in the plugin.yaml file.
7. Each part of the form should be populated with the current value found in the plugin.yaml file
8. Under each form field, I want you to add the value you find in the plugin.yaml.example so the user knows what the defaults where for each field.
9. I want there to be a save button at the bottom of each form.
10. When the user clicks the save button, they should get a modal informing them that if they save the changes, the api server will automatically restart and any current processing may be lost
11. If the user confirms the modal, then the new values should be saved in yaml format back into the original plugin.yaml file for the selected plugin
12. Example plugin.yaml file:
```
name: "audio_transcription_local"
version: "1.0.0"
enabled: true
dependencies: ["noise_reduction"]
config:
  output_directory: "data/transcripts_local"
  model_name: "base.en"
  max_concurrent_tasks: 4
  clean_up_transcript: true   
```

## .env settings

At the top of the Settings Page accordion, I want you to add a section for configuring the .env file.  This section should behave exactly like the other sections, with a form field for each variable in the .env file for the FastAPI server (found in ../).  Also use the .env.example file to find the defaults for each variable to populate the form as above.

## Consistent components

Try to use consistent components if you can to handle the interactions.  Think about how you can reuse components so there is less code and the code is more managable.



