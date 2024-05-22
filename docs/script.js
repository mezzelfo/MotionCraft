get = id => document.getElementById(id);

function author_node(author) {
    var span = document.createElement("span");
    var a = document.createElement("a");
    var sup = document.createElement("sup");
    a.textContent = author.name;
    // If the author email contains a "@" symbol, make it a mailto link
    if (author.email.includes("@"))
        a.href = "mailto:" + author.email;
    else
        a.href = author.email;
    sup.textContent = author.footnote.map(String).join(",");
    sup.textContent += author.affiliations.map(String).join(",");
    span.appendChild(a);
    span.appendChild(sup);
    return span
}

function affiliations_node(affiliations) {
    var span = document.createElement("span");
    span.innerHTML = affiliations.map((affiliation, index) =>
        "<sup>" + (index + 1).toString() + "</sup>" + affiliation
    ).join(", ");
    return span
}

function footnote_node(footnotes) {
    var span = document.createElement("span");
    // footnotes is a list of pairs of the form [symbol, footnote]
    // Then make a string of the form "<sup>symbol</sup> footnote"
    // Then join the strings with ", "
    span.innerHTML = footnotes.map(footnote =>
        "<sup>" + footnote[0] + "</sup>" + footnote[1]
    ).join(", ");
    return span
}

function make_site(paper) {
    document.title = paper.title;
    get("title").textContent = paper.title;
    get("conference").textContent = paper.conference;

    // Randomly swap the first two authors
    if (Math.random() < 0.5) {
        var temp = paper.authors[0];
        paper.authors[0] = paper.authors[1];
        paper.authors[1] = temp;
    }
    
    paper.authors.map((author, index) => {
        node = author_node(author);
        get("author-list").appendChild(node);
        if (index == paper.authors.length - 1) return;
        node.innerHTML += ", "
    })
    get("affiliation-list").appendChild(affiliations_node(paper.affiliations));
    get("footnote-list").appendChild(footnote_node(paper.footnotes));
    get("abstract").textContent = paper.abstract;

    // Populate the button list with the URLs from the paper
    buttonlist = get("button-list");
    for (var button in paper.URLs) {
        node = document.createElement("a");
        node.href = paper.URLs[button];

        img = document.createElement("img");
        img.src = "assets/logos/arXiv.svg";
        node.appendChild(img);

        span = document.createElement("span");
        span.textContent = button;
        node.appendChild(span);

        buttonlist.appendChild(node);
    }

    // Create the citation node at the end of the page in the bibtex div
    // and add a copy button to the bibtex div
    bibtex = get("bibtex");
    bibtextext = document.createElement("div");
    bibtextext.id = "bibtex-text";
    bibtextext.textContent = atob(paper.base64bibtex);
    var button = document.createElement("button");
    button.id = "copy-button";
    button.textContent = "Copy";
    button.onclick = () => {
        var range = document.createRange();
        range.selectNode(bibtextext);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
        document.execCommand("copy");
        window.getSelection().removeAllRanges();
    }
    bibtex.appendChild(button);
    bibtex.appendChild(bibtextext);
}

function set_slider(root) {
    const slidesContainer = root.querySelector(".slides-container");
    const slide = root.querySelector(".slide");
    const prevButton = root.querySelector(".slide-arrow-prev");
    const nextButton = root.querySelector(".slide-arrow-next");
    nextButton.addEventListener("click", (event) => {
        const slideWidth = slide.clientWidth;
        slidesContainer.scrollLeft += slideWidth;
    });
    prevButton.addEventListener("click", () => {
        const slideWidth = slide.clientWidth;
        slidesContainer.scrollLeft -= slideWidth;
    });
}
function create_videos() {
    // For each video in assets/videos create a video element
    // add them to the videostrailer div
    videostrailer = document.getElementById("videostrailer")

    const videolist = [
        "assets/videos/birds_MotionCraft.mp4",
        "assets/videos/dragons_MotionCraft.mp4",
        "assets/videos/earth_MotionCraft.mp4",
        "assets/videos/glass_MotionCraft.mp4",
        "assets/videos/meltingman_MotionCraft.mp4",
        "assets/videos/satellite_MotionCraft.mp4",
        // "assets/videos/birds_T2V0.mp4",
        // "assets/videos/dragons_T2V0.mp4",
        // "assets/videos/earth_T2V0.mp4",
        // "assets/videos/glass_T2V0.mp4",
        // "assets/videos/meltingman_T2V0.mp4",
        // "assets/videos/satellite_T2V0.mp4",
    ]

    for (var videopath of videolist) {
        var video = document.createElement("video");
        video.src = videopath;
        video.autoplay = true;
        video.loop = true;
        video.muted = true;
        video.controls = true;
        video.width = 300;
        video.height = 300;
        // set video speed to that is takes 4 seconds to play the video
        video.onloadedmetadata = function () {
            this.playbackRate = this.duration / 4;
        }
        videostrailer.appendChild(video);
    }

}


fetch("./paper.json").then(response => response.json()).then(json => make_site(json));

sliders = document.getElementsByClassName("slider-wrapper")
for (var i = 0; i < sliders.length; i++) set_slider(sliders[i])

create_videos();