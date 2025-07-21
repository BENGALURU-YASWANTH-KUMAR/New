# AI Assistant Hub: UI/UX Specification

This document outlines the design, user interface (UI), and user experience (UX) specifications for the AI Assistant Hub web application.

## 1. Overview

The AI Assistant Hub is a modern web application that provides users with access to several AI-powered tools: a chatbot, a book recommender, and a Twitter sentiment analyzer. The design prioritizes a clean, minimalist, and user-friendly interface with a focus on intuitive navigation and clear feedback.

## 2. Branding & Tone

- **Tone**: Friendly, helpful, and professional.
- **Microcopy**: Clear, concise, and encouraging (e.g., “Secure & private login”, “Search books...”, “Analyze sentiment”).
- **Emojis**: Used sparingly to add warmth and personality without cluttering the interface.

## 3. Global UI/UX Standards

### 3.1. Responsive Design

The application must be fully responsive and provide an optimal viewing experience across a wide range of devices, from mobile phones to desktops.

- **Mobile**: Readable text, tappable buttons (minimum 44x44px target size), and vertical layouts.
- **Desktop**: Multi-column layouts where appropriate, taking advantage of the larger screen real estate.

### 3.2. Consistent Styling (Component Library)

A consistent visual language will be used throughout the app.

- **Fonts**: A clean, modern, and readable font family (e.g., Inter, Lato).
- **Colors**: A defined primary and secondary color palette. Colors will be used consistently for buttons, links, and sentiment indicators.
- **Buttons**: Consistent styling for primary, secondary, and tertiary buttons (size, padding, border-radius, hover/active states).
- **Component Library**: A documented library of reusable components (buttons, cards, inputs, modals) will be created to ensure consistency.

### 3.3. Motion & Animation

Subtle animations will be used to enhance the user experience and provide feedback.

- **Sidebar**: Smooth expand/collapse animation.
- **Chat Widget**: Smooth open/close animation for the chat panel.
- **Button Clicks**: Subtle visual feedback on click/tap.
- **Page Transitions**: Simple fade-in/fade-out transitions between screens.

### 3.4. Feedback Loops

The application will provide clear feedback to the user for all actions.

- **Loading States**: Spinners or skeleton loaders will be displayed while data is being fetched.
- **Error States**: User-friendly error messages (toasts or inline) will be shown when something goes wrong.
- **Success States**: Confirmation messages for successful actions (e.g., form submission).
- **Undo/Back**: Options to undo actions or navigate back where appropriate.

### 3.5. Transparency

A banner stating “Powered by AI, Gemini v2.0” will be displayed to build user trust.

## 4. Screens & Components

### 4.1. Authentication Screen

- **Layout**: A full-screen modal overlaying the homepage.
- **Components**:
  - Email input field with a label.
  - Password input field with a label and a show/hide toggle.
  - “Sign In” button (primary action).
  - “Sign in with Google” button (secondary action).
- **States**:
  - **Default**: All fields are empty.
  - **Loading**: After clicking “Sign In” or “Sign in with Google”, a spinner is displayed, and the form is disabled.
  - **Error**: If login fails, an error message is displayed.
- **Microcopy**: “Secure & private login” displayed prominently.

### 4.2. Homepage Dashboard

- **Layout**: A clean, minimalist dashboard with a hero section and feature cards.
- **Hero Section**: A welcoming message and a brief introduction to the app's features.
- **Feature Cards**: Three distinct cards for:
  - **✨ Chatbot (Gemini AI)**: Icon, brief description, “Go” button.
  - **📚 Book Recommender**: Icon, brief description, “Go” button.
  - **🐦 Twitter Sentiment Analyzer**: Icon, brief description, “Go” button.
- **Background**: A subtle background gradient or light illustration motif to give a branded feel.

### 4.3. Collapsible Sidebar Navigation

- **Layout & Behavior**: A vertical sidebar on the left. It is collapsed by default (icons-only) and expands on hover to show text labels.
- **Items**:
  - Chatbot
  - Books
  - Sentiment
  - Profile
  - Settings
  - Logout
- **Animation**: Smooth expand and collapse transition.

### 4.4. Chatbot Widget

- **Initial State**: A small, brand-colored floating bubble in the bottom-right corner of the screen.
- **Expanded State**: Clicking the bubble expands it into a full chat panel.
- **Features**:
  - **Conversational UI**: Messages displayed in speech bubbles, with user and AI avatars and timestamps.
  - **Typing Indicator**: A typing indicator is shown when the AI is generating a response.
  - **Quick Replies**: Buttons for common prompts like “Summarize,” “Translate,” and “Explain code.”
  - **Citations**: Source links or citations for factual responses.
  - **Greeting**: The first message from the AI is “Hi! What can I help you with today?”

### 4.5. Book Recommendation Screen

- **Layout**: A search bar and filter panel at the top, with a grid of result cards below.
- **Components**:
  - **Search Bar**: For searching by title or author.
  - **Filter Panel**: Filters for genre, length, and mood.
  - **Book Cards**: Displaying book cover, title, author, and star rating.
  - **“Recommend similar” links** on each card.
- **States**:
  - **Loading**: A skeleton loader is displayed while recommendations are being fetched.

### 4.6. Twitter Sentiment Screen

- **Layout**: An input field for a Twitter handle or keyword, a sentiment score chart, and a preview of recent tweets.
- **Components**:
  - **Input Field**: To enter a Twitter handle or keyword.
  - **Sentiment Chart**: A bar chart showing the percentage of positive, neutral, and negative sentiment, with corresponding colors.
  - **Tweet Preview**: A list of recent tweets, with each tweet flagged with a colored label indicating its sentiment.

## 5. Component List (Figma-Ready)

- **Atoms**:
  - Colors (Primary, Secondary, Accent, Neutral, Success, Warning, Error)
  - Typography (Headings, Body, Labels)
  - Icons (24x24 set for navigation and actions)
  - Spacing & Grid System
- **Molecules**:
  - Buttons (Primary, Secondary, Google Sign-In, Icon-only)
  - Input Fields (Text, Password)
  - Checkboxes & Toggles
  - Avatars
  - Badges/Tags (for sentiment)
  - Spinners & Loaders
- **Organisms**:
  - Feature Card (Icon, Title, Description, Button)
  - Book Card (Image, Title, Author, Rating, Link)
  - Tweet Card (Avatar, Handle, Text, Sentiment Tag)
  - Navigation Bar (Collapsed & Expanded states)
  - Chat Bubble & Chat Panel
  - Search & Filter Bar
  - Sentiment Chart
- **Templates**:
  - Login Modal
  - Dashboard Layout
  - Content Page Layout (with sidebar)

## 6. Wireframe Structure

1.  **Login Flow**
    -   User lands on the page -> Auth Modal is displayed.
    -   User enters credentials -> Clicks "Sign In".
    -   Loading spinner is shown.
    -   On success -> Redirect to Homepage Dashboard.
    -   On failure -> Error message is shown on the modal.

2.  **Homepage**
    -   **Header**: App Logo.
    -   **Sidebar (Left, Collapsed)**: Icons for navigation.
    -   **Main Content Area**:
        -   Hero section with a greeting.
        -   Grid of 3 Feature Cards (Chatbot, Books, Sentiment).
    -   **Footer**: “Powered by AI” banner.
    -   **Floating Widget (Bottom-Right)**: Chat bubble.

3.  **Content Pages (Books/Sentiment)**
    -   **Header**: App Logo.
    -   **Sidebar (Left, Collapsed)**: Active page is highlighted.
    -   **Main Content Area**: Page-specific content (e.g., Book search and results).
    -   **Footer**: “Powered by AI” banner.
    -   **Floating Widget (Bottom-Right)**: Chat bubble.

4.  **Chat Interaction**
    -   User clicks on the chat bubble.
    -   Chat panel slides/fades in.
    -   User types a message or clicks a quick reply.
    -   Typing indicator is shown while the bot replies.
    -   New message from the bot appears in the panel.