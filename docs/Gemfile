source "https://rubygems.org"

gem "jekyll", "~> 4.3"
gem "just-the-docs", "0.8.2"
gem "webrick", "~> 1.8"  # Required for Ruby 3.0+

# Pin the SCSS converter to the 2.x line (sassc/libsass); the 3.x dart-sass line
# fails on some CI Ruby versions. Matches tangent/ds.
gem "jekyll-sass-converter", "~> 2.0"

group :jekyll_plugins do
  gem "jekyll-seo-tag"
  gem "jekyll-github-metadata"
  gem "jekyll-include-cache"
  gem "jekyll-sitemap"
  gem "jekyll-remote-theme"  # Required for remote_theme on GitHub Pages
end

# Windows / JRuby do not ship zoneinfo files.
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
