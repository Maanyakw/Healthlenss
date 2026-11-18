import "./globals.css";
import Shell from "../components/Shell";

export const metadata = { title: "HealthLens | AI Health Companion", description: "Explainable AI diabetes-risk screening (academic project)." };

const THEME_SCRIPT = `try{var t=localStorage.getItem("hl3_theme");document.documentElement.dataset.theme=(t==="midnight"||t==="meadow")?t:"clinic"}catch(e){document.documentElement.dataset.theme="clinic"}`;

export default function RootLayout({ children }) {
  return (
    <html lang="en" data-theme="clinic" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Figtree:wght@400;500;600;700;800&display=swap" />
      </head>
      <body><Shell>{children}</Shell></body>
    </html>
  );
}
