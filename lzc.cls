%! Class = CLASS_NAME
%! Author = 32186
%! Date = 2024/8/24

\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{lzc.cls}[32186's Document Class]

\LoadClass{ctexart}

\RequirePackage{amsmath}
\RequirePackage{lmodern}
\RequirePackage{geometry}
\RequirePackage[hidelinks, colorlinks=true, linkcolor=red, urlcolor=blue]{hyperref}
\RequirePackage{titlesec}
\RequirePackage{xcolor}
\RequirePackage{fontenc}
\RequirePackage[outputdir=../out]{minted}
\RequirePackage{graphicx}
\RequirePackage{float}

\setmainfont{Consolas}

% 设置页边距
\geometry{
    a4paper,
    top=25.4mm, bottom=25.4mm,
    left=20mm, right=20mm,
    headheight=2.17cm,
    headsep=4mm,
    footskip=12mm
}

% 定义标题页命令
\newcommand{\maketitlepage}{
    \begin{titlepage}
    \pagestyle{empty}

    % 垂直居中
    \centering
    \vspace*{2in}

    {\Huge \bfseries \@title} \\[1.5cm]
    {\Large \@author} \\[0.5cm]
    {\large \@course} \\[0.5cm]
    {\Large \@date}

    \vfill

    \end{titlepage}
}

% 设置标题、作者、日期和课程
\renewcommand{\title}[1]{\gdef\@title{#1}}
\renewcommand{\author}[1]{\gdef\@author{#1}}
\renewcommand{\date}[1]{\gdef\@date{#1}}
\newcommand{\course}[1]{\gdef\@course{#1}}

% 设置标题颜色
\definecolor{myblue}{RGB}{59,122,183}
\newcommand{\newsection}{第\ \zhnumber{\arabic{section}}\ 部分}
\titleformat{\section}{\color{myblue}\normalfont\Large\bfseries}{\newsection}{1em}{}
\titleformat{\subsection}{\color{myblue}\normalfont\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\color{myblue}\normalfont\normalsize\bfseries}{\thesubsubsection}{1em}{}


% 定义 \picture 命令
\newcommand{\myPicture}[3][\textwidth]{
    \begin{figure}[H]
    \centering
    \includegraphics[width=#1]{#2}
    \caption{#3}
    \label{fig:#2}
    \end{figure}
}
