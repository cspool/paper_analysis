# **D.2 Example 2: Django Model Choice Field Validation Tests**

The following truncated example demonstrates tests for Django's ModelChoiceField validation with proper setup and teardown code. This test was generated to address the issue in [Django PR #13933,](https://github.com/django/django/pull/13933) which improves error messages in ModelChoiceField validation.

```
from django . forms import ModelChoiceField , ModelMultipleChoiceField
from django . db import models
from django . core . exceptions import ValidationError
from django . test import TestCase
from django . db import connection
# Create a simple test model
class TestModel ( models . Model ) :
    name = models . CharField ( max_length =100)
    def __str__ ( self ) :
         return self . name
    class Meta :
         app_label = 'test_app '
def setup_test_environment () :
    # Create the test table
    with connection . schema_editor () as schema_editor :
         schema_editor . create_model ( TestModel )
    # Create some test data
    TestModel . objects . create ( name =" Option 1")
    TestModel . objects . create ( name =" Option 2")
    TestModel . objects . create ( name =" Option 3")
def cleanup_test_environment () :
```

```
# Drop the test table
    with connection . schema_editor () as schema_editor :
         schema_editor . delete_model ( TestModel )
def test_modelchoice_validation () :
    try :
        # Test Case 1: Basic invalid choice for ModelChoiceField
         field = ModelChoiceField ( queryset = TestModel . objects .all () )
        try :
             field . clean (999) # Non - existent ID
             print (" Test Case 1: Issue resolved ")
         except ValidationError as e :
             if " 999 " not in str( e ) : # Value should be in error message
                 print (" Test Case 1: Issue reproduced ")
             else :
                 print (" Test Case 1: Issue resolved ")
        # ... [4 more test cases omitted for brevity ] ...
        # Test Case 6: Valid choice
        obj = TestModel . objects . first ()
        try :
             result = field . clean ( obj .id)
             if result == obj :
                 print (" Test Case 6: Issue resolved ")
             else :
                 print (" Test Case 6: Issue reproduced ")
         except ValidationError :
             print (" Test Case 6: Issue reproduced ")
    except Exception as e :
         print ( f" Unexpected error : {e}")
```

Listing 8: Test cases for Django ModelChoiceField validation (truncated). Most test cases error due to unhandled exceptions and do not distinguish.

