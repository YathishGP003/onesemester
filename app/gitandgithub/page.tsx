import type { Metadata } from "next";
import Sidebar from "../components/sidebar";

export const generateMetadata = (): Metadata => {
  return {
    title: "Git",
    description: "Collection of miscellaneous resources.",
    twitter: {
      card: "summary_large_image",
      title: "Github actions",
      description: "Collection of miscellaneous resources.",
      images: ["https://i.imgur.com/wPZN4AU.png"],
    },
    openGraph: {
      siteName: "One Semester",
      title: "Github",
      description: "Collection of miscellaneous resources.",
      images: [
        {
          url: "https://i.imgur.com/wPZN4AU.png",
          width: 800,
          height: 600,
        },
        {
          url: "https://i.imgur.com/wPZN4AU.png",
          width: 1800,
          height: 1600,
          alt: "One Semester Programming Roadmaps",
        },
      ],
    },
  };
};

export default function Gitandgithub() {
  return (
    <div className="flex flex-col min-h-screen items-center justify-center relative">
      <div className="flex w-full md:w-1/2 py-16">
        <Sidebar />
        <main className="flex-1 p-0 md:p-0 w-full md:pl-0 border-l border-neutral-700 ">
          <h1 className="text-3xl md:text-4xl font-bold mb-2 px-4 pt-2">
            Git and GitHub
          </h1>
          <p className="text-2xl px-4">
            <a
              href="https://git-scm.com"
              className="text-emerald-400 underline underline-offset-4"
            >
              Git
            </a>{" "}
            is the distributed{" "}
            <a
              href="https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control"
              className="text-emerald-400 underline underline-offset-4"
            >
              version control system (VCS).
            </a>{" "}
            Nearly every developer in the world uses it to manage their code. It
            has quite a monopoly on VCS. Developers use Git to:
          </p>
          <ul className="list-disc text-2xl ml-4 px-4 py-2">
            <li>Keep a history of their code changes</li>
            <li>Revert mistakes made in their code</li>
            <li>Collaborate with other developers</li>
            <li>Make backups of their code</li>
          </ul>
          <p className="text-2xl px-4 text-emerald-400 mt-4">
            Some resources to follow
          </p>
          <ul className="list-disc text-2xl ml-4 px-4 py-2">
            <li>
              <a
                href="https://git-scm.com/book/en/v2"
                className="text-emerald-400 underline underline-offset-4"
              >
                Git Book
              </a>
            </li>
            <li>
              <a
                href="https://docs.github.com/en/get-started/start-your-journey/about-github-and-git"
                className="text-emerald-400 underline underline-offset-4"
              >
                GitHub Docs
              </a>
            </li>
            <li>
              <a
                href="https://www.atlassian.com/git/tutorials"
                className="text-emerald-400 underline underline-offset-4"
              >
                Atlassian Tutorials
              </a>
            </li>
            <li>
              <a
                href="https://www.theodinproject.com/lessons/foundations-setting-up-git"
                className="text-emerald-400 underline underline-offset-4"
              >
                Basic
              </a>{" "}
              and{" "}
              <a
                href="https://www.theodinproject.com/lessons/javascript-a-deeper-look-at-git"
                className="text-emerald-400 underline underline-offset-4"
              >
                Deeper
              </a>
            </li>
            <li>
              <a
                href="https://learngitbranching.js.org/?locale=en_US"
                className="text-emerald-400 underline underline-offset-4"
              >
                Practical experience
              </a>
            </li>
          </ul>
          <p className="text-2xl px-4 text-emerald-400 mt-4">
            Youtube Git Tutorial
          </p>
          <ul className="list-disc text-2xl ml-4 px-4 py-2">
            <li>
              Programming with Mosh's{" "}
              <a
                href="https://www.youtube.com/watch?v=8JJ101D3knE&t=35s"
                className="text-emerald-400 underline underline-offset-4"
              >
                Git Tutorial
              </a>
            </li>
            <li>
              ThePrimeagen's{" "}
              <a
                href="https://www.youtube.com/watch?v=rH3zE7VlIMs"
                className="text-emerald-400 underline underline-offset-4"
              >
                Git
              </a>{" "}
              hands-on.
            </li>
            <li>
              <a
                href="https://www.youtube.com/watch?v=zTjRZNkhiEU"
                className="text-emerald-400 underline underline-offset-4"
              >
                freeCodeCamp
              </a>{" "}
              full Course for Beginners
            </li>
          </ul>
        </main>
      </div>
    </div>
  );
}
