# YoppyChat - Whop App Store Integration Roadmap

This document outlines the necessary steps to refactor and prepare the YoppyChat application for submission to the Whop App Store. The goal is to support both existing direct users and new users from Whop communities.

---

### Phase 1: Backend Refactoring & Database Migration (Completed)

This phase focused on upgrading the database and core backend logic to support the new, more complex requirements.

- [x] **Design New Database Schema:** Define tables for `profiles`, `channels`, `communities`, `user_channels`, etc., to support both user types.
- [x] **Deploy New Schema to Supabase:** Run the SQL script to create the new tables and set up Row Level Security (RLS).
- [x] **Create Centralized DB Utilities (`utils/db_utils.py`):** Consolidate all database interactions into a single, manageable module.
- [x] **Refactor Channel Submission (`/channel` route):** Update the endpoint in `app.py` to use the new DB functions for checking and creating channels.
- [x] **Refactor Background Worker (`tasks.py`):** Update `process_channel_task` to accept a `channel_id` and update the channel's status in the database.
- [x] **Refactor Data-Fetching Functions:** Update helper functions like `get_user_channels()` to query the new database schema.

---

### Phase 2: Whop Authentication & User Handling (Completed)

This phase implemented the core logic for getting Whop users into our system.

- [x] **Update `whop_auth.py`:**
    - [x] Differentiate between an **App Install** (owner) and a **Member Login** by checking for `community_id` in the OAuth callback.
    - [x] Handle the **App Install Flow**: When an owner installs, create a new entry in the `communities` table.
    - [x] Handle the **Member Login Flow**: When a member logs in, identify their community and create or update their record in the `profiles` table.
- [x] **Implement Whop License Check:**
    - [x] Create a function in `utils/subscription_utils.py` to call the Whop `validate_license` API endpoint.
    - [x] Use this function during member login to set the `is_whop_pro_member` flag in their profile.
- [x] **Unify Session Management:** Ensure the `session` object in Flask is populated consistently for both direct users and Whop users.

---

### Phase 3: Core Logic & Feature Gating (Completed)

This phase implemented the business rules that distinguish between different user tiers.

- [x] **Create Route Protection Decorator:** Build a new decorator (`@limit_enforcer`) in `utils/subscription_utils.py` that checks a user's plan (both direct and Whop) and enforces query limits.
- [x] **Apply Decorator to Protected Routes:** Add the new decorator to the `/stream_answer` route in `app.py`.
- [x] **Implement "Default Channel" for Owners:**
    - [x] Create a new setup page/flow (`/setup-community`) for community owners to configure their community's default channel after installing the app.
- [x] **Update Frontend UI:**
    - [x] Pass user status (e.g., `is_pro_member`, `is_owner`) from Flask to the Jinja2 templates.
    - [x] Use `{% if %}` blocks in the templates to conditionally show/hide the "Add Channel" button and display an "Upgrade to Pro" prompt.
- [x] **Fix Frontend Bugs:** Resolved issues with login popups, streaming answers, and template rendering.

---

### Phase 4: Testing & Deployment (In Progress)

- [x] **Thoroughly Test All User Flows:**
    - [x] Direct user signup and login.
    - [ ] Whop community owner app installation.
    - [ ] Whop free member login and usage (with limits).
    - [ ] Whop pro member login and usage.
- [ ] **Configure Whop Dashboard:**
    - [ ] Set up your app in the Whop developer dashboard.
    - [ ] Create the necessary products for owner plans and the "Pro Member" upgrade.
- [ ] **Prepare for Deployment:**
    - [ ] Update `.env` with all necessary Whop Product IDs and API keys.
    - [ ] Verify `Dockerfile` and `docker-compose.yml` are production-ready.

---

### Phase 5: Whop App Store Submission

- [ ] **Review Whop Submission Guidelines:** Read the official documentation for any final requirements.
- [ ] **Prepare App Store Assets:** Create an app icon, description, and screenshots.
- [ ] **Submit for Review.** 🎉