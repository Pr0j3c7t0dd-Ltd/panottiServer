import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import QueryProvider from "@/components/providers/QueryProvider";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Panotti - Your Private Audio Assistant",
  description: "Capture, Process, Innovate with Panotti. Your private audio assistant.",
  keywords: ["audio", "privacy", "AI", "assistant", "transcription", "video conferencing"],
  authors: [{ name: "Pr0j3ctTodd Ltd" }],
  metadataBase: new URL('https://panotti.io'),
  openGraph: {
    title: "Panotti - Your Private Audio Assistant",
    description: "Capture, Process, Innovate with Panotti. Your private audio assistant.",
    type: "website",
    locale: "en_US",
    url: "https://panotti.io",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased relative min-h-screen`}
        suppressHydrationWarning
      >
        <QueryProvider>
          {children}
        </QueryProvider>
      </body>
    </html>
  );
}
