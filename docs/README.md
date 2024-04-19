## How the blog posts are generated

We use [minima](https://github.com/jekyll/minima) theme for github pages so our pages would look like a blog feed.
The pages are configured via `_config.yml` file. Besides everything else, the config contains a very important field
called `description`, the content of this field is used by search robots (Google search for example).

[Minima](https://github.com/jekyll/minima) theme allows a lot of other customizations to make the pages more live 
([links to other media resources in the footer](https://github.com/jekyll/minima?tab=readme-ov-file#social-networks),
[enabling google analytics](https://github.com/jekyll/minima?tab=readme-ov-file#enabling-google-analytics) and a lot more).

## Current hacks

Due to historical reasons, we make a redirection from `index.html` -> `gh_page1.html` (the first blog post that was published).
The "main page" is then considered to be `home.html`. Unfortunately, `minima` hardcodes the address of the home page to always point
to `index.html`, meaning that the "home" button at the top of the page will always lead to `index.html -> gh_page1.html`. In order
to fix that, we had to modify the default header layout
([`_includes/header.html`](https://github.com/dchigarev/modin_perf_examples/blob/master/docs/_includes/header.html))
so the ["site-title" ref would point to `/home.html` instead](https://github.com/dchigarev/modin_perf_examples/blob/de74cbb8c6b37ec90725362aad0ec1df28976f94/docs/_includes/header.html#L7).

## How to add a new post

1. Create a markdown file in the `_posts` directory.
2. Name the file like `year-month-day-name-of-the-article.md`.
3. Put at the top of the created md file the following content:
    ```
    ---
    layout: post
    title: "Your title to be shown on the home screen"
    categories: misc
    permalink: /desired_path_to_the_page.html
    author: Author's name
    ---
    ```
4. Write your post in the file

## How to run the gh pages locally for debugging

1. Install ruby <b>with dev kit</b>: https://rubyinstaller.org/downloads/
2. Install dependencies: `gem install jekyll bundler`
3. Change your dir to `docs`
4. Run `bundler install`
5. Run the server: `bundle exec jekyll serve`
6. After that, you can access the pages via `127.0.0.1:4000` in your browser.
