{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
   {% if item != "__init__" %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}
