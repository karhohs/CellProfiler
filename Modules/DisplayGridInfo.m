function handles = DisplayGridInfo(handles)

% Help for the Display Grid Information module:
% Category: Other
%
% SHORT DESCRIPTION:
% Displays text information on grid (i.e. gene names).
% *************************************************************************
%
% This module will display text information in a grid pattern. It requires
% that you define a grid earlier in the pipeline using the DefineGrid
% module and also load text information using the LoadText module. This
% module allows you to load multiple sets of text data. The different sets
% can be displayed in different colors. The text information must have the
% same number of entries as there are grid locations (grid squares).
%
% See also DefineGrid and LoadText.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the grid you defined?
%infotypeVAR01 = gridgroup
GridName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = On what image would you like to display text information?
%infotypeVAR02 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What is the first set of text information that you would like to display in the grid pattern (will be red)?
%infotypeVAR03 = datagroup
DataName1 = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What is the second set of text information that you would like to display in the grid pattern (will be green)?
%choiceVAR04 = /
%infotypeVAR04 = datagroup
DataName2 = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = What is the third set of text information that you would like to display in the grid pattern (will be blue)?
%choiceVAR05 = /
%infotypeVAR05 = datagroup
DataName3 = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieve grid info from previously run module.
GridInfo = handles.Pipeline.(['Grid_' GridName]);
Rows = GridInfo.Rows;
Columns = GridInfo.Columns;
YSpacing = GridInfo.YSpacing;
XSpacing = GridInfo.XSpacing;
VertLinesX = GridInfo.VertLinesX;
VertLinesY = GridInfo.VertLinesY;
HorizLinesX = GridInfo.HorizLinesX;
HorizLinesY = GridInfo.HorizLinesY;
SpotTable = GridInfo.SpotTable;
GridXLocations = GridInfo.GridXLocations;
GridYLocations = GridInfo.GridYLocations;
YLocations = GridInfo.YLocations;
XLocations = GridInfo.XLocations;
LeftOrRight = GridInfo.LeftOrRight;
TopOrBottom = GridInfo.TopOrBottom;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieves the display image.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName);

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Activates the appropriate figure window.
FigHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
%%% Usually this image should be fairly large, so we are pretending it's a
%%% 2x2 figure window rather than 1x1.
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber);
end

CPimagesc(OrigImage,handles);
title(['Cycle #', num2str(handles.Current.SetBeingAnalyzed),', with text info displayed'])
line(VertLinesX,VertLinesY);
line(HorizLinesX,HorizLinesY);
title(['Cycle #', num2str(handles.Current.SetBeingAnalyzed), ' with text info displayed']);
set(findobj(FigHandle,'type','line'),'color',[.15 1 .15])

%%% Sets the location of Tick marks.
set(gca, 'XTick', GridXLocations(1,:)+floor(XSpacing/2))
set(gca, 'YTick', GridYLocations(:,1)+floor(YSpacing/2))

%%% Sets the Tick Labels.
if strcmp(LeftOrRight,'Right')
    set(gca, 'XTickLabel',fliplr(1:Columns))
else
    set(gca, 'XTickLabel',{1:Columns})
end
if strcmp(TopOrBottom,'Bottom')
    set(gca, 'YTickLabel',{fliplr(1:Rows)})
else
    set(gca, 'YTickLabel',{1:Rows})
end

GridLineCallback = [...
    'button = gco;'...
    'if strcmp(get(button,''String''),''Hide Grid''),'...
    'set(button,''String'',''Show Grid'');'...
    'set(get(button,''UserData''),''visible'',''off'');'...
    'else,'...
    'set(button,''String'',''Hide Grid'');'...
    'set(get(button,''UserData''),''visible'',''on'');'...
    'end;'];

uicontrol(FigHandle,...
    'Units','normalized',...
    'Position',[.05 .02 .1 .04],...
    'String','Hide Grid',...
    'BackgroundColor',[.7 .7 .9],...
    'FontSize',handles.Preferences.FontSize,...
    'UserData',findobj(FigHandle,'type','line'),...
    'Callback',GridLineCallback);

Colors = {'Yellow' 'Magenta' 'Cyan' 'Red' 'Green' 'Blue' 'White' 'Black'};

GridLineColorCallback = [...
    'Colors = get(gcbo,''string'');'...
    'Value = get(gcbo,''value'');'...
    'set(get(gcbo,''UserData''),''color'',Colors{Value});'];

uicontrol(FigHandle,...
    'Units','normalized',...
    'Style','popupmenu',...
    'Position',[.16 .02 .1 .04],...
    'String',Colors,...
    'BackgroundColor',[.7 .7 .9],...
    'FontSize',handles.Preferences.FontSize,...
    'Value',1,...
    'UserData',findobj(FigHandle,'type','line'),...
    'Callback',GridLineColorCallback);

if ~strcmp(DataName1,'/')
    Text1 = handles.Measurements.Image.(DataName1);
    Description1 = handles.Measurements.Image.([DataName1 'Description']);

    temp=reshape(SpotTable,1,[]);
    tempText = Text1;
    for i=1:length(temp)
        Text1{i} = tempText{temp(i)};
    end

    TextHandles1 = text(XLocations,YLocations+floor(YSpacing/3),Text1,'Color','red','fontsize',handles.Preferences.FontSize);

    ButtonCallback = [...
        'button = gco;'...
        'if strcmp(get(button,''String''),''Hide Text1''),'...
        'set(button,''String'',''Show Text1'');'...
        'set(get(button,''UserData''),''visible'',''off'');'...
        'else,'...
        'set(button,''String'',''Hide Text1'');'...
        'set(get(button,''UserData''),''visible'',''on'');'...
        'end;'];

    uicontrol(FigHandle,...
        'Units','normalized',...
        'Position',[.27 .02 .1 .04],...
        'String','Hide Text1',...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'UserData',TextHandles1,...
        'Callback',ButtonCallback);

    Text1ColorCallback = [...
        'Colors = get(gcbo,''string'');'...
        'Value = get(gcbo,''value'');'...
        'set(get(gcbo,''UserData''),''color'',Colors{Value});'];

    uicontrol(FigHandle,...
        'Units','normalized',...
        'Style','popupmenu',...
        'Position',[.38 .02 .1 .04],...
        'String',Colors,...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'Value',4,...
        'UserData',TextHandles1,...
        'Callback',GridLineColorCallback);
end

if ~strcmp(DataName2,'/')
    Text2 = handles.Measurements.Image.(DataName2);
    Description2 = handles.Measurements.Image.([DataName2 'Description']);

    temp=reshape(SpotTable,1,[]);
    tempText = Text2;
    for i=1:length(temp)
        Text2{i} = tempText{temp(i)};
    end

    TextHandles2 = text(XLocations,YLocations+2*floor(YSpacing/3),Text2,'Color','green','fontsize',handles.Preferences.FontSize);

    ButtonCallback = [...
        'button = gco;'...
        'if strcmp(get(button,''String''),''Hide Text2''),'...
        'set(button,''String'',''Show Text2'');'...
        'set(get(button,''UserData''),''visible'',''off'');'...
        'else,'...
        'set(button,''String'',''Hide Text2'');'...
        'set(get(button,''UserData''),''visible'',''on'');'...
        'end;'];


    uicontrol(FigHandle,...
        'Units','normalized',...
        'Position',[.49 .02 .1 .04],...
        'String','Hide Text2',...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'UserData',TextHandles2,...
        'Callback',ButtonCallback);

    Text2ColorCallback = [...
        'Colors = get(gcbo,''string'');'...
        'Value = get(gcbo,''value'');'...
        'set(get(gcbo,''UserData''),''color'',Colors{Value});'];

    uicontrol(FigHandle,...
        'Units','normalized',...
        'Style','popupmenu',...
        'Position',[.6 .02 .1 .04],...
        'String',Colors,...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'Value',5,...
        'UserData',TextHandles2,...
        'Callback',GridLineColorCallback);
end

if ~strcmp(DataName3,'/')
    Text3 = handles.Measurements.Image.(DataName3);
    Description3 = handles.Measurements.Image.([DataName3 'Description']);

    temp=reshape(SpotTable,1,[]);
    tempText = Text3;
    for i=1:length(temp)
        Text3{i} = tempText{temp(i)};
    end

    TextHandles3 = text(XLocations,YLocations+YSpacing,Text3,'Color','blue','fontsize',handles.Preferences.FontSize);

    ButtonCallback = [...
        'button = gco;'...
        'if strcmp(get(button,''String''),''Hide Text3''),'...
        'set(button,''String'',''Show Text3'');'...
        'set(get(button,''UserData''),''visible'',''off'');'...
        'else,'...
        'set(button,''String'',''Hide Text3'');'...
        'set(get(button,''UserData''),''visible'',''on'');'...
        'end;'];

    uicontrol(FigHandle,...
        'Units','normalized',...
        'Position',[.71 .02 .1 .04],...
        'String','Hide Text3',...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'UserData',TextHandles3,...
        'Callback',ButtonCallback);

    Text1ColorCallback = [...
        'Colors = get(gcbo,''string'');'...
        'Value = get(gcbo,''value'');'...
        'set(get(gcbo,''UserData''),''color'',Colors{Value});'];

    uicontrol(FigHandle,...
        'Units','normalized',...
        'Style','popupmenu',...
        'Position',[.82 .02 .1 .04],...
        'String',Colors,...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'Value',6,...
        'UserData',TextHandles3,...
        'Callback',GridLineColorCallback);
end