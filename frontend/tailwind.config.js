/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "var(--color-primary)",
        "primary-hover": "var(--color-primary-hover)",
        "on-primary": "var(--color-on-primary)",
        success: "var(--color-success)",
        warning: "var(--color-warning)",
        error: "var(--color-error)",
        background: "var(--color-background)",
        surface: "var(--color-surface)",
        border: "var(--color-border)",
        foreground: "var(--color-text-primary)",
        muted: "var(--color-text-secondary)",
        code: "var(--color-code)",
        "code-text": "var(--color-code-text)",
        "warning-bg": "var(--color-warning-bg)",
        "warning-border": "var(--color-warning-border)",
        "warning-text": "var(--color-warning-text)",
        "info-bg": "var(--color-info-bg)",
        "info-border": "var(--color-info-border)",
        "info-text": "var(--color-info-text)",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        card: "0 12px 32px rgb(15 23 42 / 0.08)",
      },
    },
  },
  plugins: [],
};
