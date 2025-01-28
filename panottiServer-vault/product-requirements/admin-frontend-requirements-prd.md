# PRD: FastAPI Admin Frontend with Next.js (v14 App Router)

## Table of Contents

1. [Overview](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#overview)
2. [Goals and Objectives](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#goals-and-objectives)
3. [Assumptions & Constraints](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#assumptions--constraints)
4. [User Roles & Flows](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#user-roles--flows)
5. [Features & Requirements](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#features--requirements)
    1. [Basic Admin Login](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#basic-admin-login)
    2. [Admin Password Management](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#admin-password-management)
    3. [Settings Page](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#settings-page)
        1. [Environment Variables Management](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#environment-variables-management)
        2. [Plugins Management](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#plugins-management)
6. [Technical Design](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#technical-design)
    1. [Project Structure](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#project-structure)
    2. [Data Flow](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#data-flow)
    3. [Next.js 14 (App Router) Best Practices](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#nextjs-14-app-router-best-practices)
    4. [API Routes / Route Handlers](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#api-routes--route-handlers)
7. [UI/UX & Wireframe Concepts](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#uiux--wireframe-concepts)
8. [Deployment & Delivery](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#deployment--delivery)
9. [Testing & Validation](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#testing--validation)
10. [Appendix: Implementation Details](https://chatgpt.com/c/67980e9b-2030-8000-bc04-239697eb7b24#appendix-implementation-details)

---

## 1. Overview

The FastAPI Admin Frontend is a **Next.js 14** application located in `/admin-frontend` within the same repository as a Python-based FastAPI server (root directory). The goal of the admin frontend is to:

- Provide secure login access (basic admin authentication)
- Allow administrators to view and modify application settings (including `.env` variables and plugin settings)
- Control the FastAPI server (restarting it when necessary and viewing logs)

This document outlines the **product requirements** for the new features to be implemented in the Next.js application using the **App Router**.

---

## 2. Goals and Objectives

1. **Security**: Restrict application access via a basic admin login with a hashed password
2. **Configurability**: Offer a user-friendly UI for managing environment variables and plugin settings
3. **Maintainability**: Ensure consistent UI components and modular code that is easy to extend
4. **Reliability**: Changes to configurations should properly persist, and the FastAPI server should restart as needed

---

## 3. Assumptions & Constraints

1. **Default Password**: The default password is `Pa55w0rd`, hashed and stored in a file at the root of the repository
2. **Technology Stack**:
    - **Frontend**: Next.js 14 (App Router, React-based)
    - **Backend**: FastAPI (Python)
3. **File Structure**:
    - Root directory hosts the **FastAPI** server and a password hash file
    - The Next.js app is inside `/admin-frontend`
    - Plugins are in `../app/plugins`, each with a `plugin.yaml` and `plugin.yaml.example`
    - Environment variables are located in `../.env` and `../.env.example`
4. **Server Control**: Modifying settings via the Next.js UI can restart the FastAPI server. This requires:
    - An API (route handler) call that triggers a server restart
    - The UI to warn the user of potential data loss
5. **Plugin Configuration Files**:
    - The plugin `.yaml` file is the source of truth for the plugin's current config
    - Each plugin also has a `.yaml.example` with defaults and documentation

---

## 4. User Roles & Flows

### Admin User

1. **Login Flow**:
    
    - Admin lands on the login page (e.g., `/login`)
    - Enters password
    - If it matches the stored hash (default or updated), access is granted
    - If it's the default password, the user is prompted to change it immediately via a modal
2. **Changing Password**:
    
    - Admin navigates to the "Admin" menu or is prompted upon first login
    - Provides new password (twice for confirmation)
    - UI updates the password hash file in the repository
3. **Managing Settings**:
    
    - Admin navigates to "Settings" page (e.g., `/admin/settings`)
    - Reviews `.env` variables and plugin configurations
    - Updates values, clicks "Save" → sees a modal confirming the server restart
    - Confirms and triggers the new settings to be written
    - FastAPI server restarts automatically

---

## 5. Features & Requirements

### 1. Basic Admin Login

**Description**  
Implement a basic authentication flow in the Next.js app:

- Use a hashed password stored in a file (e.g., `password-hash.txt`) in the root folder of the Next.js admin app
- The default password is `Pa55w0rd`
- Users must enter the correct password (hashed and compared to stored hash) to access any protected pages

**User Stories**

1. As an admin, I want to be prompted for a password before accessing any admin features so that unauthorized users cannot access sensitive settings
2. As an admin, I want my password stored securely in a hashed format so that plain text passwords are never exposed

**Functional Requirements**

- **FR1**: If the admin is not authenticated, they are redirected to `/login`
- **FR2**: Compare user-entered password (post-hash) against the stored hash to authorize access
- **FR3**: Store default hash for `Pa55w0rd` in a root file; ability to update this file with a new hash

**Acceptance Criteria**

- User cannot bypass or view any protected route without valid authentication
- Storing hashed password file does not expose the password in plain text

---

### 2. Admin Password Management

**Description**  
Add a new "Admin" menu section and a prompt to ensure the user changes the default password on first login.

**User Stories**

1. As an admin, I want a dedicated menu for changing the admin password so I can update it regularly
2. As an admin, I want to be forced to change the default password on first use to maintain security best practices

**Functional Requirements**

- **FR4**: On first login with default credentials, system triggers a modal prompting the user to change the password immediately
- **FR5**: "Admin" menu includes a component to change the password (enter old password, enter new password twice)
- **FR6**: UI updates the stored password hash file after successful password change

**Acceptance Criteria**

- When logging in with the default hash, a modal forces password change
- Admin can successfully change the password from the "Admin" menu any time

---

### 3. Settings Page

The Settings Page includes two major sections: **Environment Variables** and **Plugins**. Both follow a similar pattern of reading current values and defaults, letting the user modify them, and saving updates in the appropriate files.

#### 3.1. Environment Variables Management

**Description**  
At the top of the Settings Page, display an accordion section for **.env** variables. Each variable in `../.env` should be editable. Show defaults from `../.env.example`.

**User Stories**

1. As an admin, I want to see all environment variables configured in the `.env` file
2. As an admin, I want to know what the default values are (`.env.example`) so that I can revert or compare as needed
3. As an admin, I want to save updated values, triggering a server restart to apply changes immediately

**Functional Requirements**

- **FR7**: Parse `../.env` for existing variables and `../.env.example` for default values
- **FR8**: Display each variable in a text input with the current value from `.env`. Show the default below the input
- **FR9**: "Save" button triggers a modal warning the user about server restart. Upon confirmation, the updated `.env` is written, and the server restarts

**Acceptance Criteria**

- Correct variables and default values are displayed in the accordion
- Saving changes updates the `.env` file and restarts the server

#### 3.2. Plugins Management

**Description**  
Below the Environment Variables section, display multiple accordions for each plugin found in `../app/plugins`. Each plugin has a `plugin.yaml` and a `plugin.yaml.example`, containing:

```yaml
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

**User Stories**

1. As an admin, I want to manage each plugin independently, toggling it on/off and updating its config parameters
2. As an admin, I want to see the plugin's default values (from `plugin.yaml.example`) so I understand recommended settings
3. As an admin, I want the plugin name displayed in a user-friendly way (e.g., removing underscores and sentence-casing)

**Functional Requirements**

- **FR10**: Find and list all plugin directories in `../app/plugins`
- **FR11**: Parse both `plugin.yaml` and `plugin.yaml.example` for each plugin
- **FR12**: Create an accordion section for each plugin:
    - Title the accordion section with the plugin name, removing underscores and converting to sentence case (e.g., "Audio transcription local")
    - A checkbox for the `enabled` field
    - A text input for each `config` key, showing the current value from `plugin.yaml`. Below each text input, display the default value from `plugin.yaml.example`
- **FR13**: A "Save" button triggers a modal warning of a server restart
- **FR14**: Confirming the modal saves new values to `plugin.yaml` and restarts the server

**Acceptance Criteria**

- Each plugin's YAML data is read and displayed correctly
- Saving changes updates the correct plugin's `plugin.yaml` file
- Server restarts upon saving changes

---

## 6. Technical Design

### 1. Project Structure

A recommended structure for the repository (simplified). Note how we use the `app/` directory in Next.js 14:

```
root/  
├─ .env                   # Env file for FastAPI
├─ .env.example          # Default env values for FastAPI
├─ app/
│  ├─ main.py            # FastAPI entry point
│  ├─ plugins/
│  │  ├─ pluginA/
│  │  │  ├─ plugin.yaml
│  │  │  ├─ plugin.yaml.example
│  │  └─ pluginB/
│  │     ├─ plugin.yaml
│  │     ├─ plugin.yaml.example
├─ admin-frontend/
│  ├─ app/
│  │  ├─ (auth)/
│  │  │  └─ login/
│  │  │     └─ page.tsx       # Login page
│  │  ├─ (protected)/
│  │  │  ├─ admin/
│  │  │  │  └─ page.tsx       # Admin page (password mgmt)
│  │  │  └─ settings/
│  │  │     └─ page.tsx       # Settings page
│  │  ├─ api/
│  │  │  ├─ login/
│  │  │  │  └─ route.ts       # Route handler for login
│  │  │  ├─ change-password/
│  │  │  │  └─ route.ts       # Route handler for password change
│  │  │  ├─ env/
│  │  │  │  └─ route.ts       # Route handler for reading/updating .env
│  │  │  └─ plugins/
│  │  │     ├─ route.ts       # GET all plugins
│  │  │     └─ [pluginName]/
│  │  │        └─ route.ts    # POST updates to a specific plugin
│  │  ├─ layout.tsx           # Global layout for Next.js App Router
│  │  ├─ global.css
│  │  └─ page.tsx             # Default home page or redirect
│  ├─ components/
│  │  ├─ Accordion.tsx   
│  │  ├─ Modal.tsx       
│  │  ├─ Form.tsx        
│  │  └─ ...
│  ├─ hooks/
│  ├─ lib/
│  │  └─ api.ts          # Client-side helpers for calling route handlers
│  ├─ public/
│  ├─ styles/
│  ├─ password-hash.txt  # Stores the hashed admin password
│  ├─ next.config.js
│  └─ package.json
```

#### Notes on App Router

- **`app/layout.tsx`**: Defines root layout, global imports, metadata, and error handling boundaries
- **Segment Folders**: `(auth)` or `(protected)` can be used to group related pages behind shared layout or middleware checks
- **`app/api/*`**: **Route Handlers** replace older `pages/api/*.ts` and provide a cleaner file-based API approach

---

### 2. Data Flow

1. **Login**
    
    - User enters password on the `app/(auth)/login/page.tsx`
    - A client-side request is made to `app/api/login/route.ts`
    - The route handler validates the password (hashed input) vs. `password-hash.txt`
2. **Authentication**
    
    - If valid, store a session token (HTTP-only cookie, JWT, or other mechanism)
    - If invalid, return error message
    - The Next.js 14 application can use **Middleware** or a higher-level layout to restrict access to `(protected)` routes
3. **Reading/Writing Configuration**
    
    - Next.js calls an API route handler (e.g., `app/api/env/route.ts`) to:
        - Read `.env` / `.env.example`
        - Enumerate plugins, read `plugin.yaml` & `plugin.yaml.example`
    - When saving changes, the route handler writes to the relevant YAML or `.env` file, then triggers a server restart request to FastAPI
4. **Server Restart**
    
    - Upon confirming changes, a request is sent to the FastAPI server to gracefully restart, or an external script/command is triggered to handle the restart

---

### 3. Next.js 14 (App Router) Best Practices

1. **Route Handlers**
    
    - Create route handlers in `admin-frontend/app/api/.../route.ts` for server-side logic (reading/writing files, hashing passwords, etc.)
    - Keep secure logic in server-only code to avoid exposing sensitive operations to the client
2. **Layouts & Middleware**
    
    - Use a top-level `layout.tsx` for global UI structure (header, footer)
    - Use a `(protected)/layout.tsx` to add authentication checks (via Next.js Middleware or server components) for pages in the `(protected)` folder
3. **Server Components vs. Client Components**
    
    - For pages that read data at build or request time and do not need client interactivity, use **Server Components**
    - For interactive forms (plugins settings, .env edits), use **Client Components**. Fetch data on the server side via route handlers or server components, then pass it down
4. **State Management**
    
    - Minimal global state is required. Rely on direct calls to route handlers.
    - Keep any local state for forms in client components
5. **Handling Sensitive Files**
    
    - Keep `password-hash.txt` and environment variables inaccessible to the public by placing them outside of `app/` or in a location that only server code can read

---

### 4. API Routes / Route Handlers

Below are potential **App Router** route handlers. They reside in `admin-frontend/app/api/.../route.ts`.

1. **`POST /api/login`**
    
    - **File**: `admin-frontend/app/api/login/route.ts`
    - **Description**: Validates password by comparing hashed input with stored hash (`password-hash.txt`)
    - **Request Body**: `{ password: string }`
    - **Response**: `{ success: boolean, message?: string, requiresPasswordChange?: boolean }`
2. **`POST /api/change-password`**
    
    - **File**: `admin-frontend/app/api/change-password/route.ts`
    - **Description**: Updates `password-hash.txt` with a new hash
    - **Request Body**: `{ oldPassword: string, newPassword: string }`
    - **Response**: `{ success: boolean, message?: string }`
3. **`GET /api/env`** (and `POST /api/env`)
    
    - **File**: `admin-frontend/app/api/env/route.ts`
    - **Description**:
        - **GET**: Reads `.env` and `.env.example`, returning combined data
        - **POST**: Writes updated values to `.env` and triggers server restart
    - **GET Response**: `{ env: Record<string, string>, defaults: Record<string, string> }`
    - **POST Request Body**: `{ env: Record<string, string> }`
    - **POST Response**: `{ success: boolean, message?: string }`
4. **`GET /api/plugins`**
    
    - **File**: `admin-frontend/app/api/plugins/route.ts`
    - **Description**: Reads all plugin directories, returning arrays of data for each plugin (`plugin.yaml`, `plugin.yaml.example`)
    - **Response**:
        
        ```json
        [
          {
            "name": "audio_transcription_local",
            "friendlyName": "Audio transcription local",
            "enabled": true,
            "config": {...},
            "defaults": {...}
          },
          ...
        ]
        ```
        
5. **`POST /api/plugins/[pluginName]`**
    
    - **File**: `admin-frontend/app/api/plugins/[pluginName]/route.ts`
    - **Description**: Writes updated values to a single plugin's `plugin.yaml` file, triggers server restart
    - **Request Body**:
        
        ```json
        {
          "enabled": true,
          "config": {
            "output_directory": "...",
            ...
          }
        }
        ```
        
    - **Response**: `{ success: boolean, message?: string }`

---

## 7. UI/UX & Wireframe Concepts

1. **Login Page** (`/login`)
    
    - Simple form with password field
    - "Login" button; displays error message on invalid password
2. **Home/Index Page** (`/`)
    
    - Could redirect to either a dashboard or `/login` if not authenticated
    - Future enhancements: logs, server status, etc.
3. **Admin Page** (`/admin`)
    
    - Menu for changing the admin password (old password, new password, confirm)
    - Prompt on first login if default password is detected
4. **Settings Page** (`/admin/settings`)
    
    - **Accordion** with two main sections:
        1. `.env Settings`
            - Fields for each variable, default displayed below
            - "Save" triggers the restart modal
        2. **Plugins**
            - An accordion for each plugin (title with sentence-cased name)
            - Toggle checkbox for `enabled`
            - Form fields for each config key, with default values shown below
            - "Save" triggers a confirm modal before server restart

---

## 8. Deployment & Delivery

1. **Frontend Build**
    - Standard Next.js 14 build (`npm run build`)
    - Deployed to your hosting/Vercel or served via Node process on the same server as FastAPI
2. **Backend Deployment**
    - FastAPI is deployed in a Python environment (Docker, systemd, or other)
    - Ensure the Next.js route handlers can communicate with FastAPI for server restart
3. **Security**
    - Make sure `password-hash.txt` is **not** publicly accessible
    - Use environment variables for sensitive data in production

---

## 9. Testing & Validation

1. **Unit Tests**
    - For route handlers (reading/writing `.env`, `plugin.yaml`, etc.)
    - For password hashing and comparison
2. **Integration Tests**
    - Verify login flow works end to end (wrong vs. right password)
    - Test saving environment variables and plugins triggers restart properly
3. **Manual QA**
    - Confirm default password triggers a forced password change modal
    - Confirm environment variable changes persist and appear after a server restart

---

## 10. Appendix: Implementation Details

- **Password Hashing**: Use a library like `bcrypt` or `argon2` in Node for hashing. On the FastAPI side, the same library or Python equivalents can be used if needed for cross-checking.
- **Server Restart Mechanism**:
    - Could be a simple script accessible via an endpoint in FastAPI (e.g., `/api/restart`).
    - Or, run a Docker container that can be restarted with Docker commands from the Next.js route handler or a webhook.
- **Plugin Discovery**:
    - Possibly read the directory listing of `../app/plugins` using Node’s `fs` module in the route handler.
- **Error Handling**:
    - For security, avoid returning detailed stack traces for errors in production.
    - Show user-friendly error messages in the UI.

---

### Final Notes

This PRD details a Next.js 14 **App Router** approach to building a secure admin interface for a FastAPI backend. The structure, route handlers, and UI flows should provide a solid foundation for you to implement the required environment and plugin management features, along with secure admin login and password management.