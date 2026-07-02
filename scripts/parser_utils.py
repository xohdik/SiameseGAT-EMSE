"""Tree-sitter parser utilities for 9 programming languages"""

from tree_sitter import Parser, Language

# Lazy imports to avoid requiring all packages at once
_imports = {}

def _import_python():
    import tree_sitter_python
    return tree_sitter_python

def _import_java():
    import tree_sitter_java
    return tree_sitter_java

def _import_javascript():
    import tree_sitter_javascript
    return tree_sitter_javascript

def _import_ruby():
    import tree_sitter_ruby
    return tree_sitter_ruby

def _import_go():
    import tree_sitter_go
    return tree_sitter_go

def _import_php():
    import tree_sitter_php
    return tree_sitter_php

def _import_c():
    import tree_sitter_c
    return tree_sitter_c

def _import_cpp():
    import tree_sitter_cpp
    return tree_sitter_cpp

def _import_rust():
    import tree_sitter_rust
    return tree_sitter_rust

# Language loader functions
_LANGUAGE_LOADERS = {
    'python': lambda: Language(_import_python().language()),
    'java': lambda: Language(_import_java().language()),
    'javascript': lambda: Language(_import_javascript().language()),
    'ruby': lambda: Language(_import_ruby().language()),
    'go': lambda: Language(_import_go().language()),
    'php': lambda: Language(_import_php().language_php()),
    'c': lambda: Language(_import_c().language()),
    'cpp': lambda: Language(_import_cpp().language()),
    'rust': lambda: Language(_import_rust().language()),
}

# Cache for language objects
_LANGUAGE_CACHE = {}

# Cache for parser objects
_PARSER_CACHE = {}

def get_language(lang_name):
    """Get a Language object for the specified language"""
    if lang_name not in _LANGUAGE_LOADERS:
        raise ValueError(f"Unsupported language: {lang_name}. "
                       f"Supported: {list(_LANGUAGE_LOADERS.keys())}")
    
    if lang_name not in _LANGUAGE_CACHE:
        _LANGUAGE_CACHE[lang_name] = _LANGUAGE_LOADERS[lang_name]()
    
    return _LANGUAGE_CACHE[lang_name]

def get_parser(lang_name):
    """Get a Parser object for the specified language"""
    if lang_name not in _PARSER_CACHE:
        language = get_language(lang_name)
        _PARSER_CACHE[lang_name] = Parser(language)
    
    return _PARSER_CACHE[lang_name]

def parse_code(lang_name, code_bytes):
    """Parse code in the specified language"""
    parser = get_parser(lang_name)
    return parser.parse(code_bytes)

def get_supported_languages():
    """Get list of supported languages"""
    return list(_LANGUAGE_LOADERS.keys())

def test_all_parsers():
    """Test all parsers"""
    test_code = {
        "python": b"x = 1\nif x > 0:\n    y = x + 1",
        "java": b"class A { void f() { int x = 1; if (x > 0) { int y = x + 1; } } }",
        "cpp": b"#include <iostream>\nint main() { int x = 1; if (x > 0) { int y = x + 1; } }",
        "c": b"#include <stdio.h>\nint main() { int x = 1; if (x > 0) { int y = x + 1; } return 0; }",
        "javascript": b"let x = 1;\nif (x > 0) { let y = x + 1; }",
        "ruby": b"x = 1\nif x > 0\n  y = x + 1\nend",
        "go": b"package main\nfunc main() { x := 1; if x > 0 { y := x + 1; _ = y } }",
        "php": b"<?php $x = 1; if ($x > 0) { $y = $x + 1; } ?>",
        "rust": b"fn main() { let x = 1; if x > 0 { let y = x + 1; } }",
    }
    
    print("Testing all parsers:\n")
    
    for lang in get_supported_languages():
        if lang in test_code:
            try:
                tree = parse_code(lang, test_code[lang])
                n = len(tree.root_node.children)
                print(f"  {lang:<12}: ✓ parsed ({n} top-level nodes)")
            except ImportError:
                print(f"  {lang:<12}: ✗ Package not installed")
            except Exception as e:
                print(f"  {lang:<12}: ✗ {e}")
        else:
            print(f"  {lang:<12}: ✗ No test code")
    
    print(f"\nSupported languages: {', '.join(get_supported_languages())}")

# Helper function for your build_graphs.py
def get_node_text(node, source_bytes):
    """Get text of a node from source bytes"""
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8')

def find_nodes_by_type(tree, node_type, max_results=100):
    """Find all nodes of a specific type in the AST"""
    nodes = []
    
    def _traverse(node):
        if len(nodes) >= max_results:
            return
        if node.type == node_type:
            nodes.append(node)
        for child in node.children:
            _traverse(child)
    
    _traverse(tree.root_node)
    return nodes

def get_function_nodes(tree, lang_name):
    """Get function/method nodes based on language"""
    if lang_name == 'python':
        return find_nodes_by_type(tree, 'function_definition')
    elif lang_name in ['java', 'c', 'cpp']:
        return find_nodes_by_type(tree, 'function_declarator')
    elif lang_name == 'javascript':
        return find_nodes_by_type(tree, 'function_declaration')
    elif lang_name == 'go':
        return find_nodes_by_type(tree, 'function_declaration')
    elif lang_name == 'ruby':
        return find_nodes_by_type(tree, 'method')
    elif lang_name == 'php':
        return find_nodes_by_type(tree, 'function_definition')
    elif lang_name == 'rust':
        return find_nodes_by_type(tree, 'function_item')
    else:
        return []

if __name__ == "__main__":
    test_all_parsers()