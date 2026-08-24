# Meta-Agent

- 1. Call get\_tools\_in\_category('Education')
- 2. Call get\_tools\_in\_category('Other')
- 3. Call create\_agent\_category\_level('Education') **→** Create Category-Agent ('Education')
- 4. Call get\_tools\_in\_category('Text Analysis')
- 5. Call create\_agent\_category\_level('Text Analysis') **→** Create Category-Agent ('Text Analysis')
- 6. Call finish\_search()

- 1. Call get\_tools\_descriptions (['GetBooksInfo', 'Book Finder', 'Random Word', 'Dictionary', 'Master Dictionary', 'Random Words', 'Random Ukrainian Word', 'Random Words - Spanish and French'])
- 2. Call create\_agent\_tool\_level(['GetBooksInfo', 'Book Finder']) **→** Create Tool-Agent-1
- 3. Call create\_agent\_tool\_level(['Random Word', 'Random Words', 'Random Ukrainian Word', 'Random Words - Spanish and French']) **→** Create Tool-Agent-2
- 4. Call create\_agent\_tool\_level(['Dictionary', 'Master Dictionary']) **→** Create Tool-Agent-3
- 5. Call finish\_search()

