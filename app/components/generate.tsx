"use client";
import { useEffect, useState, useRef } from "react";
import { TaskComponent } from "./task";

const Generate = ({
  learningPlan,
  defaultMonth = 1,
  defaultWeek = 1,
}: {
  learningPlan: LearningPlan;
  defaultMonth?: number;
  defaultWeek?: number;
}) => {
  const [selectedMonth, setSelectedMonth] = useState<number>(defaultMonth);
  const [expandedMonth, setExpandedMonth] = useState<number | null>(
    defaultMonth
  );
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const currentWeekRef = useRef<number>(1);

  const lastBKeyPressTime = useRef<number>(0);

  const isInputElement = (element: HTMLElement | null): boolean => {
    if (!element) return false;
    const tagName = element.tagName.toLowerCase();
    return (
      tagName === "input" ||
      tagName === "textarea" ||
      tagName === "select" ||
      element.isContentEditable
    );
  };

  const scrollToWeek = (weekId: string) => {
    const element = document.getElementById(weekId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
    setIsSidebarOpen(false);
  };

  const moveToAdjacentWeek = (direction: "prev" | "next") => {
    const currentMonth = learningPlan.months.find(
      (m) => m.month === selectedMonth
    );
    if (!currentMonth?.weeks) return;

    const maxWeeks = currentMonth.weeks.length;
    let nextWeek = currentWeekRef.current;

    if (direction === "next" && nextWeek < maxWeeks) {
      nextWeek += 1;
    } else if (direction === "prev" && nextWeek > 1) {
      nextWeek -= 1;
    }

    if (nextWeek !== currentWeekRef.current) {
      currentWeekRef.current = nextWeek;
      scrollToWeek(`week-${selectedMonth}-${nextWeek}`);
    }
  };

  const changeMonth = (direction: "prev" | "next") => {
    const monthIndex = learningPlan.months.findIndex(
      (m) => m.month === selectedMonth
    );
    if (direction === "next" && monthIndex < learningPlan.months.length - 1) {
      setSelectedMonth(learningPlan.months[monthIndex + 1].month);
    } else if (direction === "prev" && monthIndex > 0) {
      setSelectedMonth(learningPlan.months[monthIndex - 1].month);
    }
  };

  useEffect(() => {
    if (defaultWeek) {
      scrollToWeek(`week-${defaultMonth}-${defaultWeek}`);
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      // Check if we're in an input element
      const target = event.target as HTMLElement;
      if (isInputElement(target)) {
        return;
      }

      // Handle 'bd' combination for going back to home
      if (event.key === "b") {
        lastBKeyPressTime.current = Date.now();
      } else if (event.key === "d") {
        const currentTime = Date.now();
        if (currentTime - lastBKeyPressTime.current < 300) {
          window.location.href = "/";
        }
      }

      // Prevent default for our navigation keys
      if (["{", "}", "h", "l"].includes(event.key)) {
        event.preventDefault();
      }

      // Handle week navigation
      if (event.key === "{") {
        moveToAdjacentWeek("prev");
      } else if (event.key === "}") {
        moveToAdjacentWeek("next");
      }

      // Handle month navigation
      if (event.key === "h") {
        changeMonth("prev");
        window.scrollTo({ top: 0, behavior: "smooth" });
      } else if (event.key === "l") {
        changeMonth("next");
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [defaultWeek, defaultMonth, selectedMonth]);

  useEffect(() => {
    const scrollToElement = (elementId: string, retryCount = 0) => {
      const element = document.getElementById(elementId);
      if (element) {
        setTimeout(() => {
          element.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);
      } else if (retryCount < 5) {
        setTimeout(() => {
          scrollToElement(elementId, retryCount + 1);
        }, 200);
      }
    };

    // Handle URL parameters and hash
    const handleUrlNavigation = () => {
      const urlParams = new URLSearchParams(window.location.search);
      const month = urlParams.get("m");
      const hash = window.location.hash.slice(1); // Remove the # symbol

      if (month) {
        setSelectedMonth(parseInt(month));
        setExpandedMonth(parseInt(month));
      }

      if (hash) {
        scrollToElement(hash);
      }
    };

    handleUrlNavigation();
    window.addEventListener("popstate", handleUrlNavigation);
    return () => window.removeEventListener("popstate", handleUrlNavigation);
  }, []);

  return (
    <div className="flex flex-col min-h-screen items-center justify-center relative">
      {/* Mobile Toggle Button */}
      <button
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        className="fixed top-4 right-4 z-50 md:hidden bg-emerald-700 text-white p-2 px-4"
      >
        {isSidebarOpen ? "✕" : "☰"}
      </button>

      {isSidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      <div className="flex w-full md:w-1/2 py-16">
        {/* Sidebar with mobile responsive styles */}
        <aside
          className={`
                    fixed md:relative top-0 left-0 h-full 
                    w-64 md:w-48                    text-2xl 
                    bg-neutral-950 md:bg-transparent
                    transform transition-transform duration-300 ease-in-out
                    md:translate-x-0 z-40
                    ${isSidebarOpen ? "translate-x-0" : "-translate-x-full"}
                `}
        >
          <a
            href="/"
            className={`w-full block text-left px-4 py-2 transition-color bg-transparent hover:bg-neutral-800`}
          >
            Home
          </a>

          {learningPlan.months.map((month) => (
            <div key={month.month}>
              <button
                onClick={() => {
                  setSelectedMonth(month.month);
                  setExpandedMonth(
                    expandedMonth === month.month ? null : month.month
                  );
                }}
                className={`w-full text-left px-4 py-2 transition-colors ${
                  selectedMonth === month.month
                    ? "bg-emerald-700 text-white"
                    : "bg-transparent hover:bg-neutral-800"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span>Month {month.month}</span>
                  {month.weeks && (
                    <span
                      className="transform text-sm transition-transform duration-200"
                      style={{
                        transform:
                          expandedMonth === month.month
                            ? "rotate(180deg)"
                            : "rotate(0deg)",
                      }}
                    >
                      ▼
                    </span>
                  )}
                </div>
              </button>
              {expandedMonth === month.month && month.weeks && (
                <div className="pl-4 py-1 bg-neutral-950">
                  {month.weeks.map((week) => (
                    <button
                      key={week.week}
                      onClick={() =>
                        scrollToWeek(`week-${month.month}-${week.week}`)
                      }
                      className="w-full text-left p-2 text-xl text-neutral-400 hover:text-emerald-700 transition-colors"
                    >
                      Week {week.week}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          <a
            href="/beyond"
            className="w-full block border-t-[1px] border-neutral-700 text-left  px-4 py-2 transition-color bg-transparent hover:bg-neutral-800"
          >
            Beyond
          </a>
          <a
            href="/gitandgithub"
            className="w-full block text-left  px-4 py-2 transition-color bg-transparent hover:bg-neutral-800"
          >
            Github
          </a>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-0 md:p-0 w-full md:pl-0 border-l border-neutral-700 ">
          <h1 className="text-3xl md:text-4xl font-bold mb-2 px-4 pt-2">
            {learningPlan.lesson}
          </h1>
          <div className="flex px-4 pb-4 items-center">
            <img
              src={learningPlan.author.image}
              alt={learningPlan.author.name}
              className="h-8 w-8"
            />
            <p className="text-xl px-4 text-emerald-400">
              {learningPlan.author.name}
            </p>
            <div className="flex gap-4 items-center">
              <a
                href={learningPlan.author.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-emerald-500 h-5 w-5 hover:text-emerald-600"
              >
                <svg
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                >
                  {" "}
                  <path
                    d="M5 3h6v2H5v14h14v-6h2v8H3V3h2zm8 0h8v8h-2V7h-2V5h-4V3zm0 8h-2v2H9v2h2v-2h2v-2zm4-4h-2v2h-2v2h2V9h2V7z"
                    fill="currentColor"
                  />{" "}
                </svg>
              </a>
              {learningPlan.author.github && (
                <a
                  href={learningPlan.author.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-500 h-5 w-5 hover:text-emerald-600"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    {" "}
                    <path
                      fill="currentColor"
                      d="M5 2h4v2H7v2H5V2Zm0 10H3V6h2v6Zm2 2H5v-2h2v2Zm2 2v-2H7v2H3v-2H1v2h2v2h4v4h2v-4h2v-2H9Zm0 0v2H7v-2h2Zm6-12v2H9V4h6Zm4 2h-2V4h-2V2h4v4Zm0 6V6h2v6h-2Zm-2 2v-2h2v2h-2Zm-2 2v-2h2v2h-2Zm0 2h-2v-2h2v2Zm0 0h2v4h-2v-4Z"
                    />{" "}
                  </svg>
                </a>
              )}
              {learningPlan.author.twitter && (
                <a
                  href={learningPlan.author.twitter}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-500 text-xl h-4 w-4 flex items-center justify-center text-center hover:text-emerald-600"
                >
                  <p> 𝕏 </p>
                </a>
              )}
            </div>
          </div>

          {learningPlan.months
            .filter((month) => month.month === selectedMonth)
            .map((month) => (
              <div key={month.month} className="space-y-6">
                {month.weeks ? (
                  month.weeks.map((week) => (
                    <div
                      key={week.week}
                      id={`week-${month.month}-${week.week}`}
                      className="scroll-mt-16"
                    >
                      <h3 className="text-2xl font-medium px-4">
                        <span className="text-emerald-400">##</span> Week{" "}
                        {week.week} | {week.description}
                      </h3>
                      {week.tasks.map((task, index) => (
                        <TaskComponent
                          key={`${month.month}-${week.week}-${index}`}
                          task={task}
                          monthId={month.month}
                          weekId={week.week}
                          taskIndex={index}
                          className="border-t-[1px] border-t-neutral-700"
                        />
                      ))}
                    </div>
                  ))
                ) : (
                  <div className="">
                    {month.tasks?.map((task, index) => (
                      <TaskComponent
                        key={`${month.month}-null-${index}`}
                        task={task}
                        monthId={month.month}
                        weekId={null}
                        taskIndex={index}
                        className="border-t-[1px] border-t-neutral-700"
                      />
                    ))}
                  </div>
                )}
              </div>
            ))}
        </main>
      </div>
    </div>
  );
};

export default Generate;
