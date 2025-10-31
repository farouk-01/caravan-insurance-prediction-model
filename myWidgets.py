import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Markdown, Math
import scipy.stats as sp
import pandas as pd

class AllVarComparator:
    def __init__(self, vars_to_compare, df):
        self.vars_to_compare = vars_to_compare
        self.df = df

        self.dropdown1 = widgets.Dropdown(
            options=self.vars_to_compare,
            description='Variable ...'
        )

        self.dropdown2 = widgets.Dropdown(
            options=self.vars_to_compare,
            description='Variable ...'
        )

        self.dropdown1.observe(self._on_dropdown_change, names='value')
        self.dropdown2.observe(self._on_dropdown_change, names='value')

        self.out = widgets.Output()

        self.container = widgets.VBox([
            widgets.HBox([self.dropdown1, self.dropdown2]), 
            self.out
        ])
    
    def _on_dropdown_change(self, change):
        var1 = self.dropdown1.value
        var2 = self.dropdown2.value
    
        if not hasattr(self, 'results_out'):
            self.results_out = widgets.Output()
        if not hasattr(self, 'description_out'):
            self.description_out = widgets.Output()

        with self.results_out:
            clear_output(wait=True)

        with self.description_out:
            clear_output(wait=True)
            display(Markdown(f"$A = $ {self.df.attrs['description'][var1]}, $B = $ {self.df.attrs['description'][var2]}"))

        if not hasattr(self, 'btn_hbox'):
            btn_cross = widgets.Button(description='CrossTab')
            btn_inter = widgets.Button(description="P(A n B)")
            btn_1_si_2 = widgets.Button(description="P(A|B)")
            btn_2_si_1 = widgets.Button(description="P(B|A)")
            self.btn_hbox = widgets.HBox([btn_cross, btn_inter, btn_1_si_2, btn_2_si_1])

            self.container = widgets.VBox([self.description_out, self.btn_hbox, self.results_out])
            display(self.container)

            def get_crossTab():
                var1 = self.dropdown1.value
                var2 = self.dropdown2.value
                col1 = self.df[var1]
                col2 = self.df[var2]
                crossTab = pd.crosstab(col1, col2)
                return crossTab

            def show_crossTab(btn):
                with self.results_out:
                    clear_output(wait=True)
                    display(Math('A \\cap B'))
                    display(get_crossTab().style.background_gradient(cmap='Greens'))

            def show_inter(btn):
                with self.results_out:
                    clear_output(wait=True)
                    p_inter = get_crossTab() / len(self.df)
                    display(Math('P(A \\cap B)'))
                    display(p_inter.style.background_gradient(cmap='Greens'))

            def show_1_si_2(btn):
                with self.results_out:
                    clear_output(wait=True)
                    crossTab = get_crossTab()
                    p_1_si_2 = crossTab.div(crossTab.sum(axis=0), axis=1)
                    display(Math(f'P(A|B)'))
                    display(p_1_si_2.style.background_gradient(cmap='Greens'))

            def show_2_si_1(btn):
                with self.results_out:
                    clear_output(wait=True)
                    crossTab = get_crossTab()
                    p_2_si_1 = crossTab.div(crossTab.sum(axis=1), axis=0)
                    display(Math(f'P(B|A)'))
                    display(p_2_si_1.style.background_gradient(cmap='Greens'))

            btn_cross.on_click(show_crossTab)
            btn_inter.on_click(show_inter)
            btn_1_si_2.on_click(show_1_si_2)
            btn_2_si_1.on_click(show_2_si_1)

    def __len__(self):
        return len(self.vars_to_compare)
    
    def __getitem__(self, index):
        return self.vars_to_compare[index]
    
    def get_widget(self):
        return self.container
    
#TODO : ajouter que sa affiche le premier article dès le début  
class SingleVarComparator:
    def __init__(self, vars_to_compare, var_to_be_compared, df, cmap='coolwarm', axis=0):
        self.var_to_be_compared = var_to_be_compared
        self.vars_to_compare = vars_to_compare
        self.df = df
        self.col_to_be_compared = self.df[var_to_be_compared]
        self.cmap = cmap
        self.axis = axis
        
        self.dropdown1 = widgets.Dropdown(
            options=self.vars_to_compare,
            description='Variable ...'
        )

        self.dropdown1.observe(self._on_dropdown_change, names='value')
        self.out = widgets.Output()

        self.container = widgets.VBox([
            widgets.HBox([self.dropdown1]), 
            self.out
        ])
    
    def _on_dropdown_change(self, change):
        var1 = self.dropdown1.value

        if not hasattr(self, 'results_out'):
            self.results_out = widgets.Output()
        if not hasattr(self, 'description_out'):
            self.description_out = widgets.Output()

        with self.results_out:
            clear_output(wait=True)

        with self.description_out:
            clear_output(wait=True)
            display(Markdown(f"$A = $ {self.df.attrs['description'][var1]}, $B = $ {self.df.attrs['description'][self.var_to_be_compared]}"))

        if not hasattr(self, 'btn_hbox'):
            btn_cross = widgets.Button(description='CrossTab')
            btn_inter = widgets.Button(description="P(A n B)")
            btn_1_si_2 = widgets.Button(description="P(A|B)")
            btn_2_si_1 = widgets.Button(description="P(B|A)")
            self.btn_hbox = widgets.HBox([btn_cross, btn_inter, btn_1_si_2, btn_2_si_1])

            self.container = widgets.VBox([self.description_out, self.btn_hbox, self.results_out])
            display(self.container)

            def get_crossTab():
                var1 = self.dropdown1.value
                col1 = self.df[var1]
                crossTab = pd.crosstab(col1, self.col_to_be_compared)
                return crossTab

            def show_crossTab(btn):
                with self.results_out:
                    clear_output(wait=True)
                    display(Math('A \\cap B'))
                    display(get_crossTab().style.background_gradient(cmap=self.cmap, axis=self.axis))

            def show_inter(btn):
                with self.results_out:
                    clear_output(wait=True)
                    p_inter = get_crossTab() / len(self.df)
                    display(Math('P(A \\cap B)'))
                    display(p_inter.style.background_gradient(cmap=self.cmap, axis=self.axis))

            def show_1_si_2(btn):
                with self.results_out:
                    clear_output(wait=True)
                    crossTab = get_crossTab()
                    p_1_si_2 = crossTab.div(crossTab.sum(axis=0), axis=1)
                    display(Math(f'P(A|B)'))
                    display(p_1_si_2.style.background_gradient(cmap=self.cmap, axis=self.axis))

            def show_2_si_1(btn):
                with self.results_out:
                    clear_output(wait=True)
                    crossTab = get_crossTab()
                    p_2_si_1 = crossTab.div(crossTab.sum(axis=1), axis=0)
                    display(Math(f'P(B|A)'))
                    display(p_2_si_1.style.background_gradient(cmap=self.cmap, axis=self.axis))

            btn_cross.on_click(show_crossTab)
            btn_inter.on_click(show_inter)
            btn_1_si_2.on_click(show_1_si_2)
            btn_2_si_1.on_click(show_2_si_1)

    def __len__(self):
        return len(self.vars_to_compare)
    
    def __getitem__(self, index):
        return self.vars_to_compare[index]
    
    def get_widget(self):
        return self.container

class VarComparator:
    def __init__(self, var1, var2, df, axis=0):
        self.col1 = df[var1]
        self.col2 = df[var2]
        self.df = df
        self.axis = axis
        crossTab = pd.crosstab(self.col1, self.col2)
        self.out = widgets.Output()
        with self.out:
            clear_output(wait=True)
            display(Markdown(f"{var1} = {self.df.attrs['description'][var1]}"))
            display(Markdown(f"{var2} = {self.df.attrs['description'][var2]}"))
            display(crossTab.style.background_gradient(cmap='Greens', axis=self.axis))

    def __len__(self):
        return len(self.vars_to_compare)
    
    def __getitem__(self, index):
        return self.vars_to_compare[index]
    
    def get_widget(self):
        return self.out
    
    
